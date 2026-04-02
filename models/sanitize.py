"""
Data sanitization utilities.
Reusable guards for inf/NaN, dtype enforcement, and prediction clamping.
"""

import logging

import numpy as np
import pandas as pd

from .config import PRED_CLAMP_UPPER

log = logging.getLogger("BottomUpForecast")


def sanitize_inf_nan(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Replace inf/-inf → NaN → 0.0 for specified columns.
    
    SPI logs, interaction terms, and lag merges can introduce inf/NaN
    that cause sklearn/LightGBM to raise ValueError.
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Verify
    inf_count = df[columns].apply(lambda s: np.isinf(s).sum()).sum()
    nan_count = df[columns].isna().sum().sum()
    if inf_count > 0 or nan_count > 0:
        log.warning(f"   ⚠️  Post-sanitize: {inf_count} inf, {nan_count} NaN remaining")
    else:
        log.info("   ✅ Feature sanitization complete (no inf/NaN in feature columns)")

    return df


def enforce_numeric_dtypes(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Coerce all feature columns to float64.
    
    Spark toPandas() and merge() can silently produce object dtype columns
    (mixed nulls, string residuals). sklearn/statsmodels crash on non-numeric.
    """
    non_numeric = [c for c in columns
                   if c in df.columns and not np.issubdtype(df[c].dtype, np.number)]
    if non_numeric:
        log.warning(f"   ⚠️  Non-numeric feature columns: {non_numeric} — coercing to float64")
        for col in non_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    log.info(f"   ✅ All {len(columns)} feature columns verified as numeric")
    return df


def clamp_predictions(preds: np.ndarray, lower: float = 0,
                      upper: float = PRED_CLAMP_UPPER) -> np.ndarray:
    """Replace non-finite predictions with 0 and clip to [lower, upper].
    
    Statsmodels ZINB/NB can overflow to inf when parameters are extreme
    (exp(large_number) → inf). Without clamping, these inf values propagate
    to Ridge.fit() and cause sklearn ValueError.
    """
    preds = np.where(np.isfinite(preds), preds, 0.0)
    return np.clip(preds, lower, upper)
