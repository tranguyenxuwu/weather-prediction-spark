"""
WeatherPredict — Phase 5 Predictions Dashboard
=================================================
Streamlit app that loads pre-calculated monthly cache for instant
actual vs predicted storm count visualization.

Usage:
    cd WeatherPredict
    conda activate pyspark
    streamlit run app/app.py
"""

import pickle
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

import sys

# ── Wire up the models package ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.config import (
    MODEL_DIR, CACHE_PATH, MONTH_NAMES,
)
from models.phase5_rollup import derive_features

ONI_CSV = BASE_DIR / "oni.csv"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WeatherPredict — Storm Forecast",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

MONTH_MAP = {i: MONTH_NAMES[i] for i in range(1, 13)}


# ══════════════════════════════════════════════════════════════════════════════
# Resource Initialization
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_ensemble_models():
    """Load Phase 5 ensemble model dict (one model per target)."""
    ensemble_path = MODEL_DIR / "phase5_ensemble.pkl"
    legacy_path = MODEL_DIR / "ridge_monthly_model.pkl"

    if ensemble_path.exists():
        with open(ensemble_path, "rb") as f:
            models = pickle.load(f)
        model_type = "ensemble"
    elif legacy_path.exists():
        with open(legacy_path, "rb") as f:
            models = pickle.load(f)
        model_type = "legacy"
    else:
        st.error("No trained Phase 5 model found. Run the pipeline first:\n"
                 "```\npython -m models.bottom_up_forecast --phase5\n```")
        st.stop()
    return models, model_type


@st.cache_data
def load_monthly_data():
    """Load pre-calculated monthly cache from disk."""
    if not Path(CACHE_PATH).exists():
        return None
    return pd.read_parquet(CACHE_PATH)


@st.cache_data
def load_oni():
    """Load ONI index for ENSO sidebar display."""
    if not ONI_CSV.exists():
        return None
    df = pd.read_csv(ONI_CSV, skiprows=0, names=["date", "oni"], header=0)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["oni"] = pd.to_numeric(df["oni"], errors="coerce")
    df = df[df["oni"] > -999]
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df[["year", "month", "oni"]]


@st.cache_data
def get_predictions_from_cache():
    """
    Load cached monthly data, derive features, and predict total storm counts.
    This is fast (~2s) since the cache is only ~500 rows.
    """
    monthly_data = load_monthly_data()
    if monthly_data is None:
        return None

    ensemble_dict, model_type = load_ensemble_models()

    # Use the same feature derivation as the training pipeline
    monthly_data, feature_columns = derive_features(monthly_data, legacy_mode=(model_type == "legacy"))

    # Predict total storm counts
    model = ensemble_dict.get("count")
    if model:
        X = monthly_data[feature_columns].copy()
        monthly_data["pred_count"] = np.maximum(model.predict(X), 0)

    return monthly_data





# ══════════════════════════════════════════════════════════════════════════════
# Chart Builders
# ══════════════════════════════════════════════════════════════════════════════

def build_monthly_chart(year_df, year):
    """Grouped bar chart: actual vs predicted storm counts for one year."""
    if year_df.empty:
        return None

    year_df = year_df.copy()
    year_df["month_name"] = year_df["month"].map(MONTH_MAP)
    year_df = year_df.sort_values("month")

    pred_col = "pred_count" if "pred_count" in year_df.columns else "predicted"

    actual_total = int(year_df["actual_count"].sum())
    predicted_total = round(year_df[pred_col].sum(), 1)
    valid = year_df[pred_col].notna()
    mae = round((year_df.loc[valid, "actual_count"] - year_df.loc[valid, pred_col]).abs().mean(), 2)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=year_df["month_name"], y=year_df["actual_count"],
        name="Actual", marker_color="#4285F4",
        hovertemplate="<b>%{x}</b><br>Actual: %{y:.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=year_df["month_name"], y=year_df[pred_col],
        name="Predicted", marker_color="#FF9800",
        hovertemplate="<b>%{x}</b><br>Predicted: %{y:.1f}<extra></extra>",
    ))

    y_max = max(
        year_df["actual_count"].max(),
        year_df[pred_col].max() if valid.any() else 0,
    )
    y_ceil = int(y_max) + 2

    fig.add_vrect(x0="Aug", x1="Oct", fillcolor="rgba(255,182,193,0.25)",
                  layer="below", line_width=0)
    fig.add_annotation(x="Sep", y=y_ceil - 0.3, text="<i>Peak Season</i>",
                       showarrow=False, font=dict(size=14, color="red"))
    fig.add_annotation(
        xref="paper", yref="paper", x=1, y=0,
        text=f"<i>Total: Actual={actual_total}, Predicted={predicted_total} | MAE={mae}/mo</i>",
        showarrow=False, font=dict(size=11, color="red"),
        xanchor="right", yanchor="bottom",
    )

    fig.update_layout(
        barmode="group",
        title=dict(text=f"{year} — Monthly Storm Count: Actual vs Predicted",
                   font=dict(size=18), x=0.5),
        xaxis_title="Month", yaxis_title="Storm Count",
        yaxis=dict(dtick=2, range=[0, y_ceil]),
        margin=dict(l=60, r=30, t=60, b=60),
        height=480, bargap=0.25, bargroupgap=0.08,
    )
    return fig


def build_annual_chart(monthly_data):
    """Line chart of annual actual vs predicted totals across all years."""
    if monthly_data is None or monthly_data.empty:
        return None

    annual = (
        monthly_data.groupby("year")
        .agg(actual=("actual_count", "sum"), predicted=("pred_count", "sum"))
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=annual["year"], y=annual["actual"], name="Actual",
        mode="lines+markers", line=dict(color="#4285F4", width=2),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=annual["year"], y=annual["predicted"], name="Predicted",
        mode="lines+markers", line=dict(color="#FF9800", width=2, dash="dash"),
        marker=dict(size=6),
    ))
    fig.update_layout(
        title=dict(text="Annual Storm Count: Actual vs Predicted", font=dict(size=18), x=0.5),
        xaxis_title="Year", yaxis_title="Total Storms",
        margin=dict(l=60, r=30, t=60, b=60),
        height=400,
    )
    return fig




# ══════════════════════════════════════════════════════════════════════════════
# Main UI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.title("🌀 Monthly Storm Count Forecast")

    # Load cached data
    monthly_data = get_predictions_from_cache()
    if monthly_data is None or monthly_data.empty:
        st.error(
            "No pre-calculated cache found. Run the pipeline first:\n\n"
            "```bash\n"
            "python -m models.bottom_up_forecast --prepare\n"
            "python -m models.bottom_up_forecast --phase5\n"
            "```"
        )
        st.stop()

    st.caption("Using pre-calculated Phase 5 cache — instant results")

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("# 🌀 Storm Forecast")
        st.markdown("---")

        available_years = sorted(monthly_data["year"].unique())
        year = st.selectbox("**Year**", available_years,
                            index=len(available_years) - 1)

        st.markdown("---")

        # ── ONI display ──
        oni_df = load_oni()
        if oni_df is not None:
            oni_match = oni_df[oni_df["year"] == year]
            if len(oni_match) > 0:
                oni_avg = oni_match["oni"].mean()
                if oni_avg >= 0.5:
                    phase = "🔴 El Niño"
                elif oni_avg <= -0.5:
                    phase = "🔵 La Niña"
                else:
                    phase = "⚪ Neutral"
                st.metric("ENSO Phase", phase)
                st.metric("Avg ONI", f"{oni_avg:+.2f}")

        st.markdown("---")
        _, model_type = load_ensemble_models()
        if model_type == "ensemble":
            st.caption("Model: Stacked Ensemble\n(ZINB + Tweedie LGBM + Ridge)")
        else:
            st.caption("Model: Split-Season Poisson (legacy)")

        cache_years = f"{int(monthly_data['year'].min())}–{int(monthly_data['year'].max())}"
        st.caption(f"Cache: {len(monthly_data)} months ({cache_years})")

    # ── Get data for selected year ──
    year_df = monthly_data[monthly_data["year"] == year].copy()

    if year_df.empty:
        st.warning(f"No data available for {year}.")
        return

    pred_col = "pred_count"

    # ── Metrics row ──
    actual_total = int(year_df["actual_count"].sum())
    predicted_total = round(year_df[pred_col].sum(), 1) if pred_col in year_df.columns else 0
    valid = year_df[pred_col].notna() if pred_col in year_df.columns else pd.Series(False)
    mae = round((year_df.loc[valid, "actual_count"] - year_df.loc[valid, pred_col]).abs().mean(), 2) if valid.sum() > 0 else 0.0
    peak_month = int(year_df.loc[year_df["actual_count"].idxmax(), "month"]) if year_df["actual_count"].max() > 0 else -1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Actual Storms", str(actual_total))
    c2.metric("Predicted Storms", str(predicted_total))
    c3.metric("Monthly MAE", str(mae))
    c4.metric("Peak Month", MONTH_MAP.get(peak_month, "—"))

    st.divider()

    # ── Monthly chart ──
    st.markdown("### 📊 Monthly Comparison")
    fig = build_monthly_chart(year_df, year)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # ── Annual trend ──
    if "pred_count" in monthly_data.columns:
        st.markdown("### 📈 Annual Trend")
        fig_annual = build_annual_chart(monthly_data)
        if fig_annual:
            st.plotly_chart(fig_annual, use_container_width=True)

    st.markdown("---")

    # ── Detail table ──
    with st.expander("📋 Monthly Details"):
        tbl_cols = ["month", "actual_count", pred_col, "monthly_SPI"]
        tbl = year_df[[c for c in tbl_cols if c in year_df.columns]].copy()
        tbl.insert(0, "Month", tbl["month"].map(MONTH_MAP))
        tbl = tbl.drop(columns=["month"])
        tbl = tbl.rename(columns={
            "actual_count": "Actual",
            pred_col: "Predicted",
            "monthly_SPI": "SPI",
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()