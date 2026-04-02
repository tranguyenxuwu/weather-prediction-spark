"""
Phase 5: Monthly Storm Trend Forecasting.

Decomposed into sub-functions:
- prepare_monthly_cache(): Spark inference + SPI aggregation → save cache
- load_monthly_cache(): Load pre-calculated DataFrame (no Spark needed)
- derive_features(): Lag features, ENSO interactions, encoding
- train_models(): Multithreaded 7-target model training
- evaluate(): Logging, CSV export, model save
"""

import gc
import logging
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, IntegerType, DoubleType,
)

from .config import (
    ALL_FEATURES, ALL_TARGETS, LANDFALL_TARGETS, MONTH_NAMES,
    MODEL_DIR, CACHE_PATH, PROB_THRESHOLD,
    LANDFALL_TRANSITION_PATH, STORM_LANDFALLS_PATH, FULLBASIN_PATH,
    PARQUET_DIR,
)
from .sanitize import sanitize_inf_nan, enforce_numeric_dtypes
from .ensemble import expanding_window_cv

log = logging.getLogger("BottomUpForecast")


# ══════════════════════════════════════════════════════════════════════════════
# Step 1-4: Spark-heavy work → save cache
# ══════════════════════════════════════════════════════════════════════════════

def prepare_monthly_cache(spark, lgbm_model, calibrator, train_means, full_df):
    """
    Run distributed inference (mapInPandas) on the full 225M-row dataset,
    aggregate to monthly SPI variants, join ground truth and ONI,
    and save the resulting ~500-row DataFrame as a parquet cache.

    This is the expensive Spark step. After saving, subsequent Phase 5
    runs can load the cache instantly with load_monthly_cache().
    """
    log.info("=" * 60)
    log.info("PHASE 5 PREPARE: Building monthly data cache")
    log.info("=" * 60)

    t0 = time.time()

    # ── Step 1: Distributed inference via mapInPandas ──
    log.info("   Step 1: Running calibrated inference on full dataset...")

    model_bytes = pickle.dumps(lgbm_model)
    cal_bytes   = pickle.dumps(calibrator)
    means_dict  = train_means.to_dict()
    feature_cols = list(ALL_FEATURES)

    bc_model    = spark.sparkContext.broadcast(model_bytes)
    bc_cal      = spark.sparkContext.broadcast(cal_bytes)
    bc_features = spark.sparkContext.broadcast(feature_cols)

    inference_cols = list(dict.fromkeys(["year", "month"] + ALL_FEATURES))

    log.info("   Preparing inference dataframe (applying native na.fill)...")

    inference_df = full_df.select(*inference_cols).na.fill(means_dict)

    full_df.unpersist()
    gc.collect()

    # Pre-extract monthly ONI
    log.info("   Step 1.4: Pre-extracting monthly ONI values...")
    monthly_oni_pd = (
        inference_df
        .groupBy("year", "month")
        .agg(F.avg("oni_value").alias("avg_oni"))
        .toPandas()
    )

    out_schema = StructType([
        StructField("year",              IntegerType(), True),
        StructField("month",             IntegerType(), True),
        StructField("lat",               DoubleType(),  True),
        StructField("lon",               DoubleType(),  True),
        StructField("prob_storm",        DoubleType(),  True),
        StructField("prob_above_thresh", DoubleType(),  True),
    ])

    prob_threshold = PROB_THRESHOLD  # capture for closure

    def predict_batch(iterator):
        """Apply calibrated LightGBM to each partition batch."""
        import pickle as pkl
        import numpy as _np
        _model = pkl.loads(bc_model.value)
        _cal   = pkl.loads(bc_cal.value)
        _feats = bc_features.value

        for batch_df in iterator:
            raw_probs = _model.predict(batch_df[_feats].values)
            cal_probs = _cal.transform(raw_probs)

            yield pd.DataFrame({
                "year":              batch_df["year"].values,
                "month":             batch_df["month"].values,
                "lat":               batch_df["lat"].values,
                "lon":               batch_df["lon"].values,
                "prob_storm":        cal_probs,
                "prob_above_thresh": _np.where(
                    cal_probs > prob_threshold, cal_probs, 0.0
                ),
            })

    predictions = inference_df.mapInPandas(predict_batch, schema=out_schema)

    # ── Step 1.5: Join Landfall Transition Matrix ──
    log.info("   Step 1.5: Broadcasting spatial landfall transitions...")
    if Path(LANDFALL_TRANSITION_PATH).exists():
        trans_df = spark.read.parquet(LANDFALL_TRANSITION_PATH)
        # Rename and cast to DoubleType for safe join (fixes type mismatch bug)
        trans_df = (
            trans_df
            .withColumnRenamed("lat_grid", "lat")
            .withColumnRenamed("lon_grid", "lon")
            .withColumn("lat", F.col("lat").cast(DoubleType()))
            .withColumn("lon", F.col("lon").cast(DoubleType()))
        )

        predictions = predictions.join(
            F.broadcast(trans_df), on=["lat", "lon"], how="left"
        )

        prob_cols = [c for c in trans_df.columns
                     if c.startswith("prob_") and c not in ("lat", "lon")]
        predictions = predictions.fillna(0.0, subset=prob_cols)
    else:
        log.warning("   ⚠️ Landfall transition grid not found! "
                     "Landfall targets will be missing.")
        prob_cols = []

    # ── Steps 2-3: Run SPI aggregation + ground truth loading concurrently ──
    # Ground truth (IBTrACS) is independent of inference, so we prepare it
    # while Spark processes the 225M-row inference + SPI aggregation.
    from concurrent.futures import ThreadPoolExecutor

    def _load_ground_truth():
        """Load and aggregate IBTrACS ground truth (runs in parallel)."""
        log.info("   Step 3 [thread]: Loading ground truth from IBTrACS...")
        fullbasin_df = spark.read.parquet(FULLBASIN_PATH)

        if Path(STORM_LANDFALLS_PATH).exists():
            storm_targets_df = spark.read.parquet(STORM_LANDFALLS_PATH)
            fullbasin_df = fullbasin_df.join(storm_targets_df, on="SID", how="left")

            monthly_actual_df = (
                fullbasin_df
                .groupBy("year", "month")
                .agg(
                    F.countDistinct("SID").alias("actual_count"),
                    *[
                        F.countDistinct(
                            F.when(F.col("landfall_target") == t, F.col("SID"))
                        ).alias(f"actual_{t}")
                        for t in LANDFALL_TARGETS
                    ]
                )
            )
        else:
            monthly_actual_df = (
                fullbasin_df
                .groupBy("year", "month")
                .agg(F.countDistinct("SID").alias("actual_count"))
            )
        return monthly_actual_df

    def _compute_spi():
        """Compute monthly SPI aggregation (runs in parallel)."""
        log.info("   Step 2 [thread]: Calculating monthly SPI variants...")

        agg_exprs = [
            F.sum("prob_storm").alias("monthly_SPI"),
            F.sum("prob_above_thresh").alias("monthly_SPI_thresh"),
            F.count(F.when(F.col("prob_storm") > 0.30, True)).alias("monthly_SPI_count"),
            F.avg("prob_storm").alias("monthly_SPI_density"),
            F.count("*").alias("cell_count"),
        ]
        for p_col in prob_cols:
            target_name = p_col.replace("prob_", "")
            agg_exprs.append(
                F.sum(F.col("prob_storm") * F.col(p_col)).alias(f"SPI_{target_name}")
            )

        monthly_spi_df = (
            predictions
            .groupBy("year", "month")
            .agg(*agg_exprs)
        )
        monthly_spi_df = monthly_spi_df.withColumn(
            "monthly_SPI_log", F.log1p(F.col("monthly_SPI"))
        )
        return monthly_spi_df

    log.info("   🧵 Running SPI aggregation + ground truth loading concurrently...")
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_spi = pool.submit(_compute_spi)
        future_gt = pool.submit(_load_ground_truth)
        monthly_spi_df = future_spi.result()
        monthly_actual_df = future_gt.result()

    # ── Step 4: Join → Pandas → save cache ──
    log.info("   Step 4: Joining and saving cache...")

    actual_cols = [c for c in monthly_actual_df.columns if c.startswith("actual_")]

    monthly_spark = (
        monthly_spi_df
        .join(monthly_actual_df, on=["year", "month"], how="left")
        .fillna(0, subset=actual_cols)
        .orderBy("year", "month")
    )

    monthly_data = monthly_spark.toPandas()
    monthly_data = monthly_data.merge(
        monthly_oni_pd, on=["year", "month"], how="left"
    )
    monthly_data["avg_oni"] = monthly_data["avg_oni"].fillna(0.0)

    # Save cache
    monthly_data.to_parquet(CACHE_PATH, index=False)
    log.info(f"   ✅ Cache saved: {CACHE_PATH} ({len(monthly_data)} rows)")

    # Cleanup broadcasts
    bc_model.destroy()
    bc_cal.destroy()
    bc_features.destroy()

    elapsed = time.time() - t0
    log.info(f"   Prepare complete in {elapsed:.0f}s")

    return monthly_data


def load_monthly_cache():
    """Load pre-calculated monthly data from cache. No Spark needed."""
    if not Path(CACHE_PATH).exists():
        raise FileNotFoundError(
            f"Monthly cache not found: {CACHE_PATH}\n"
            f"Run with --prepare or without --phase5 first."
        )
    monthly_data = pd.read_parquet(CACHE_PATH)
    log.info(f"   ✅ Loaded cache: {CACHE_PATH} ({len(monthly_data)} rows, "
             f"{monthly_data['year'].nunique()} years)")
    return monthly_data


# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Feature derivation (pure pandas, fast)
# ══════════════════════════════════════════════════════════════════════════════

def derive_features(monthly_data, legacy_mode=False):
    """
    Derive temporal context features from the cached monthly data.
    All lag features use merge-based construction for gap safety.

    Returns:
        monthly_data: DataFrame with all features added
        feature_columns: list of feature column names for modeling
    """
    log.info("   Step 5: Deriving temporal context features...")

    monthly_data = monthly_data.sort_values(["year", "month"]).reset_index(drop=True)

    # ── Consolidated lag merge (fixes duplicate merge bug) ──
    # Build lag keys: previous month = (year, month-1), wrapping Jan→prev Dec.
    monthly_data["_lag_year"] = monthly_data["year"].where(
        monthly_data["month"] > 1, monthly_data["year"] - 1
    ).astype(int)
    monthly_data["_lag_month"] = monthly_data["month"].where(
        monthly_data["month"] > 1, 12
    ).astype(int)

    # Build all lag-1 values in one lookup table + one merge
    lag_cols = {"year": "_lag_year", "month": "_lag_month",
                "avg_oni": "oni_lag1", "monthly_SPI": "spi_lag1"}
    if "actual_count" in monthly_data.columns:
        lag_cols["actual_count"] = "actual_count_lag1"

    _lag_src = monthly_data[list(lag_cols.keys())].rename(columns=lag_cols)
    monthly_data = monthly_data.merge(
        _lag_src, on=["_lag_year", "_lag_month"], how="left"
    )

    # Fill NaN lags
    for lag_col in ["oni_lag1", "spi_lag1", "actual_count_lag1"]:
        if lag_col in monthly_data.columns:
            monthly_data[lag_col] = monthly_data[lag_col].fillna(0.0)

    monthly_data.drop(columns=["_lag_year", "_lag_month"], inplace=True)

    # 3-month trailing averages
    monthly_data["oni_3m_avg"] = (
        monthly_data.groupby("year")["avg_oni"]
        .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
        .fillna(0.0)
    )
    monthly_data["spi_3m_avg"] = (
        monthly_data.groupby("year")["monthly_SPI"]
        .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
        .fillna(0.0)
    )

    # ── ENSO × SPI interaction features ──
    log.info("   Step 5b: ENSO × SPI interaction features...")

    monthly_data["is_elnino"] = (monthly_data["avg_oni"] >= 0.5).astype(float)
    monthly_data["is_lanina"] = (monthly_data["avg_oni"] <= -0.5).astype(float)
    monthly_data["spi_x_oni"] = monthly_data["monthly_SPI"] * monthly_data["avg_oni"]
    monthly_data["spi_x_elnino"] = monthly_data["monthly_SPI"] * monthly_data["is_elnino"]
    monthly_data["spi_x_lanina"] = monthly_data["monthly_SPI"] * monthly_data["is_lanina"]
    monthly_data["oni_abs"] = monthly_data["avg_oni"].abs()

    # Landfall SPI logs
    spi_target_cols = [c for c in monthly_data.columns
                       if c.startswith("SPI_") and not c.endswith("_log")]
    for c in spi_target_cols:
        monthly_data[c] = monthly_data[c].clip(lower=0)
        monthly_data[f"{c}_log"] = np.log1p(monthly_data[c])

    # ── Mode-specific features ──
    enso_interaction_cols = [
        "is_elnino", "is_lanina", "spi_x_oni",
        "spi_x_elnino", "spi_x_lanina", "oni_abs",
    ]

    if legacy_mode:
        monthly_data["month_cat"] = pd.Categorical(
            monthly_data["month"], categories=range(1, 13)
        )
        month_dummies = pd.get_dummies(
            monthly_data["month_cat"], prefix="month", dtype=float
        )
        monthly_data = pd.concat(
            [monthly_data.drop(columns=["month_cat"]), month_dummies], axis=1
        )

        month_cols = sorted(
            [c for c in monthly_data.columns if c.startswith("month_")]
        )
        month_interaction_cols = []
        for m_col in month_cols:
            inter_col = f"spi_x_{m_col}"
            monthly_data[inter_col] = (
                monthly_data["monthly_SPI"] * monthly_data[m_col]
            )
            month_interaction_cols.append(inter_col)

        feature_columns = (
            ["monthly_SPI", "monthly_SPI_thresh", "monthly_SPI_count",
             "monthly_SPI_density", "monthly_SPI_log",
             "avg_oni", "oni_lag1", "oni_3m_avg", "spi_lag1", "spi_3m_avg"]
            + spi_target_cols
            + enso_interaction_cols
            + month_cols
            + month_interaction_cols
        )
    else:
        monthly_data["month_sin"] = np.sin(2 * np.pi * monthly_data["month"] / 12)
        monthly_data["month_cos"] = np.cos(2 * np.pi * monthly_data["month"] / 12)
        monthly_data["spi_x_month_sin"] = (
            monthly_data["monthly_SPI"] * monthly_data["month_sin"]
        )
        monthly_data["spi_x_month_cos"] = (
            monthly_data["monthly_SPI"] * monthly_data["month_cos"]
        )

        monthly_data["cumulative_TCs_YTD"] = (
            monthly_data.groupby("year")["actual_count"]
            .transform(lambda s: s.shift(1).expanding().sum().fillna(0.0))
        )
        monthly_data["spi_momentum_1m"] = (
            monthly_data["monthly_SPI"] - monthly_data["spi_lag1"]
        )
        monthly_data["oni_momentum_3m"] = (
            monthly_data.groupby("year")["avg_oni"]
            .transform(lambda x: x.diff(1).fillna(0.0))
        )

        month_cols = ["month_sin", "month_cos", "month"]
        month_interaction_cols = ["spi_x_month_sin", "spi_x_month_cos"]

        feature_columns = (
            ["monthly_SPI", "monthly_SPI_thresh", "monthly_SPI_count",
             "monthly_SPI_density", "monthly_SPI_log",
             "avg_oni", "oni_lag1", "oni_3m_avg", "spi_lag1", "spi_3m_avg",
             "cumulative_TCs_YTD", "spi_momentum_1m", "oni_momentum_3m",
             "actual_count_lag1"]
            + spi_target_cols
            + enso_interaction_cols
            + month_cols
            + month_interaction_cols
            + [f"{c}_log" for c in spi_target_cols]
        )

    # Sanitize and enforce dtypes
    monthly_data = sanitize_inf_nan(monthly_data, feature_columns)
    monthly_data = enforce_numeric_dtypes(monthly_data, feature_columns)

    return monthly_data, feature_columns


# ══════════════════════════════════════════════════════════════════════════════
# Step 6: Model training (multithreaded)
# ══════════════════════════════════════════════════════════════════════════════

def _fit_ensemble_target(monthly_data, target_col, feature_columns,
                         train_mask, test_mask):
    """Fit a stacked ensemble for one target (thread-safe)."""
    if target_col not in monthly_data.columns:
        return None

    name = target_col.replace('actual_', '')
    log.info(f"   [Ensemble] Training {target_col} with "
             f"Expanding Window CV (1990-2019)...")

    ensemble = expanding_window_cv(
        monthly_data, 1990, 2019, target_col, feature_columns
    )

    X_test = monthly_data.loc[test_mask, feature_columns].copy()
    pred_test = ensemble.predict(X_test)
    mae = mean_absolute_error(
        monthly_data.loc[test_mask, target_col], pred_test
    )

    log.info(f"      {name:<15s} Ensemble Test MAE={mae:.2f}")
    return {"pred_test": pred_test, "mae": mae, "model": ensemble}


def _fit_legacy_target(monthly_data, target_col, feature_columns,
                       train_mask, test_mask):
    """Fit split-season Poisson models for one target (thread-safe)."""
    if target_col not in monthly_data.columns:
        return None

    name = target_col.replace('actual_', '')
    peak_months = [5, 6, 7, 8, 9, 10, 11, 12]
    offpeak_months = [1, 2, 3, 4]

    poisson_grid = {"alpha": np.logspace(-4, 4, 30)}
    m_peak = GridSearchCV(
        PoissonRegressor(max_iter=3000), poisson_grid,
        cv=3, scoring="neg_mean_absolute_error",
    )
    m_offpeak = GridSearchCV(
        PoissonRegressor(max_iter=3000), poisson_grid,
        cv=3, scoring="neg_mean_absolute_error",
    )

    # Guard: PoissonRegressor requires non-negative targets
    y_values = monthly_data[target_col].clip(lower=0)

    peak_train_idx = train_mask & monthly_data["month"].isin(peak_months)
    if peak_train_idx.sum() > 0:
        m_peak.fit(
            monthly_data.loc[peak_train_idx, feature_columns].values,
            y_values.loc[peak_train_idx].values,
        )

    offpeak_train_idx = train_mask & monthly_data["month"].isin(offpeak_months)
    if offpeak_train_idx.sum() > 0:
        m_offpeak.fit(
            monthly_data.loc[offpeak_train_idx, feature_columns].values,
            y_values.loc[offpeak_train_idx].values,
        )

    pred_test = np.zeros(test_mask.sum())
    peak_test_idx = test_mask & monthly_data["month"].isin(peak_months)
    offpeak_test_idx = test_mask & monthly_data["month"].isin(offpeak_months)

    # We need index-relative positions for the pred_test array
    test_indices = monthly_data.index[test_mask]
    if peak_test_idx.any():
        peak_pos = [i for i, idx in enumerate(test_indices)
                    if monthly_data.loc[idx, "month"] in peak_months]
        pred_test[peak_pos] = m_peak.predict(
            monthly_data.loc[peak_test_idx, feature_columns].values
        )
    if offpeak_test_idx.any():
        offpeak_pos = [i for i, idx in enumerate(test_indices)
                       if monthly_data.loc[idx, "month"] in offpeak_months]
        pred_test[offpeak_pos] = m_offpeak.predict(
            monthly_data.loc[offpeak_test_idx, feature_columns].values
        )

    pred_test = np.clip(pred_test, 0, None)
    mae = mean_absolute_error(
        monthly_data.loc[test_mask, target_col], pred_test
    )

    log.info(f"      {name:<15s} Test MAE={mae:.2f}")
    return {
        "pred_test": pred_test, "mae": mae,
        "model": {"peak": m_peak, "offpeak": m_offpeak},
    }


def train_models(monthly_data, feature_columns, incomplete_years,
                 legacy_mode=False, parallel=True):
    """
    Train all target models. Optionally multithreaded.

    Returns:
        target_results: dict of {target_name: {pred_test, mae, model}}
        models_dict: dict of {target_name: model}
        monthly_data: DataFrame with pred_* columns added
    """
    log.info("   Step 6: Training models for Total Counts + Landfalls...")

    train_mask = monthly_data["year"] <= 2019
    test_mask = (
        (monthly_data["year"] >= 2020)
        & (~monthly_data["year"].isin(incomplete_years))
    )

    targets = ALL_TARGETS
    fit_fn = _fit_legacy_target if legacy_mode else _fit_ensemble_target

    target_results = {}
    models_dict = {}

    if parallel and not legacy_mode:
        max_workers = min(len(targets), os.cpu_count() or 4)
        log.info(f"   🧵 Parallel training: {len(targets)} targets × "
                 f"{max_workers} threads")

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    fit_fn, monthly_data, f"actual_{t}",
                    feature_columns, train_mask, test_mask
                ): t
                for t in targets
            }
            for future in as_completed(futures):
                t = futures[future]
                try:
                    result = future.result()
                    if result:
                        target_results[t] = result
                        models_dict[t] = result["model"]
                except Exception as e:
                    log.error(f"   ✗ {t} failed: {e}")
    else:
        for t in targets:
            t_col = f"actual_{t}"
            result = fit_fn(
                monthly_data, t_col, feature_columns, train_mask, test_mask
            )
            if result:
                target_results[t] = result
                models_dict[t] = result["model"]

    # Assign predictions to DataFrame (main thread only — thread-safe)
    for t, res in target_results.items():
        monthly_data.loc[test_mask, f"pred_{t}"] = res["pred_test"]

    return target_results, models_dict, monthly_data, train_mask, test_mask


# ══════════════════════════════════════════════════════════════════════════════
# Step 7: Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(monthly_data, target_results, models_dict,
             train_mask, test_mask, legacy_mode=False):
    """Evaluate models and save outputs."""
    log.info(f"\n   {'=' * 60}")
    log.info(f"   EVALUATION (Test Set: "
             f"{monthly_data.loc[test_mask, 'year'].min()}-"
             f"{monthly_data.loc[test_mask, 'year'].max()})")
    log.info(f"   {'=' * 60}")

    test_eval = monthly_data[test_mask].copy()
    targets = list(target_results.keys())

    # ── Bias analysis (ensemble mode) ──
    if not legacy_mode and "count" in target_results:
        y = test_eval["actual_count"]
        y_hat = test_eval["pred_count"]

        systematic_bias = (y_hat - y).sum()
        log.info(f"   Systematic Bias: {systematic_bias:+.1f} storms")

        bias_elnino = (
            y_hat[test_eval["is_elnino"] == 1]
            - y[test_eval["is_elnino"] == 1]
        ).sum()
        bias_lanina = (
            y_hat[test_eval["is_lanina"] == 1]
            - y[test_eval["is_lanina"] == 1]
        ).sum()
        bias_neutral = (
            y_hat[(test_eval["is_elnino"] != 1) & (test_eval["is_lanina"] != 1)]
            - y[(test_eval["is_elnino"] != 1) & (test_eval["is_lanina"] != 1)]
        ).sum()

        log.info(f"      Bias during El Nino: {bias_elnino:+.1f}")
        log.info(f"      Bias during La Nina: {bias_lanina:+.1f}")
        log.info(f"      Bias during Neutral: {bias_neutral:+.1f}")

        climatology_mean = monthly_data.loc[train_mask, "actual_count"].mean()
        mae_naive = mean_absolute_error(y, np.full(len(y), climatology_mean))
        mase = (target_results["count"]["mae"] / mae_naive
                if mae_naive > 0 else float('inf'))
        log.info(f"   MASE: {mase:.3f} (< 1.0 is good)")

        try:
            from sklearn.metrics import mean_tweedie_deviance
            tw_dev = mean_tweedie_deviance(y.values, y_hat.values, power=1.5)
            log.info(f"   Tweedie Deviance (p=1.5): {tw_dev:.3f}")
        except Exception as e:
            log.warning(f"   Tweedie deviance failed: {e}")

    # ── Annual summary ──
    if "count" in target_results:
        annual_actual = test_eval.groupby("year")["actual_count"].sum()
        annual_pred = test_eval.groupby("year")["pred_count"].sum()
        annual_mae = mean_absolute_error(annual_actual, annual_pred)
        log.info(f"   Annual MAE (Total Storms): {annual_mae:.2f} storms/year")

    log.info(f"\n   Annual Predictions (Actual vs Model):")
    for year, group in test_eval.groupby("year"):
        row = group[
            [f"actual_{t}" for t in targets if t in target_results]
            + [f"pred_{t}" for t in targets if t in target_results]
        ].sum()
        log.info(f"   [{year}]")
        actual_str = " | ".join(
            [f"{t}: {int(row.get(f'actual_{t}', 0))}"
             for t in targets if f"actual_{t}" in row]
        )
        pred_str = " | ".join(
            [f"{t}: {row.get(f'pred_{t}', 0):.1f}"
             for t in targets if f"pred_{t}" in row]
        )
        log.info(f"      Actual: {actual_str}")
        log.info(f"      Preds:  {pred_str}")

    # Latest year breakdown
    latest_year = int(test_eval["year"].max())
    latest_data = test_eval[test_eval["year"] == latest_year]

    log.info(f"\n   📋 {latest_year} Monthly Breakdown:")
    log.info(f"   {'Month':<6} {'Monthly_SPI':>12} {'Actual':>8} {'Predicted':>10}")
    log.info(f"   {'─'*6} {'─'*12} {'─'*8} {'─'*10}")
    for _, row in latest_data.iterrows():
        m = int(row["month"])
        pred = row.get("pred_count", 0.0)
        log.info(f"   {MONTH_NAMES[m]:<6} {row['monthly_SPI']:>12.1f} "
                 f"{int(row['actual_count']):>8} {pred:>10.1f}")

    total_actual = int(latest_data["actual_count"].sum())
    total_pred = (latest_data["pred_count"].sum()
                  if "pred_count" in latest_data.columns else 0.0)
    log.info(f"   {'─'*6} {'─'*12} {'─'*8} {'─'*10}")
    log.info(f"   {'TOTAL':<6} {'':>12} {total_actual:>8} {total_pred:>10.1f}")

    # ── Annual roll-up ──
    if "count" in target_results:
        test_mae = target_results["count"]["mae"]
    else:
        test_mae = float("nan")

    annual_summary = (
        test_eval
        .groupby("year")
        .agg(
            annual_actual=("actual_count", "sum"),
            annual_predicted=("pred_count", "sum"),
        )
        .reset_index()
        .sort_values("year")
    )
    annual_summary["error"] = (
        annual_summary["annual_actual"] - annual_summary["annual_predicted"]
    )
    annual_mae = mean_absolute_error(
        annual_summary["annual_actual"],
        annual_summary["annual_predicted"],
    )

    log.info(f"\n   🌍 ANNUAL FORECAST SUMMARY (Test Years ≥ 2020)")
    log.info(f"   {'Year':<6} {'Annual_Actual':>14} "
             f"{'Annual_Predicted':>17} {'Error':>8}")
    log.info(f"   {'─'*6} {'─'*14} {'─'*17} {'─'*8}")
    for _, row in annual_summary.iterrows():
        log.info(f"   {int(row['year']):<6} "
                 f"{int(row['annual_actual']):>14} "
                 f"{row['annual_predicted']:>17.1f} "
                 f"{row['error']:>+8.1f}")
    log.info(f"   {'─'*6} {'─'*14} {'─'*17} {'─'*8}")
    log.info(f"   Monthly MAE : {test_mae:.2f} storms/month")
    log.info(f"   Annual MAE  : {annual_mae:.2f} storms/year")
    if legacy_mode:
        log.info(f"   Best model  : Split-Season PoissonRegressor")
    else:
        log.info(f"   Best model  : Phase 5 Stacked Ensemble (ZINB + LGBM + Ridge)")

    # ── Save outputs ──
    pred_path = str(MODEL_DIR / "monthly_predictions.csv")
    pred_cols = (["year", "month", "actual_count"]
                 + [f"pred_{t}" for t in targets if t in target_results])
    monthly_data[pred_cols].to_csv(pred_path, index=False)
    log.info(f"   Saved monthly predictions → {pred_path}")

    if legacy_mode:
        save_path = MODEL_DIR / "ridge_monthly_model.pkl"
    else:
        save_path = MODEL_DIR / "phase5_ensemble.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(models_dict, f)
    log.info(f"   Saved model dictionary → {save_path}")

    return test_mae, annual_summary, annual_mae
