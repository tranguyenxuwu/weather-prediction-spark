#!/usr/bin/env python3
"""
Generate publication-quality visualisation charts from monthly_predictions.csv.
Outputs:
  docs/chart_monthly_all_years.png   – Monthly Actual vs Predicted (all test years stacked)
  docs/chart_error_distribution.png  – Residual / error analysis
  docs/chart_country_landfall.png    – Country-level landfall predictions
  docs/chart_seasonal_pattern.png    – Seasonal climatology (actual vs predicted)
"""

import pathlib, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV  = ROOT / "models" / "monthly_predictions.csv"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFBFC",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

BLUE   = "#1976D2"
ORANGE = "#FF9800"
RED    = "#E53935"
GREEN  = "#43A047"
PURPLE = "#8E24AA"
TEAL   = "#00897B"

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

COUNTRY_COLS = {
    "pred_PH": ("Philippines (PH)", "#E53935"),
    "pred_VN": ("Vietnam (VN)",     "#1976D2"),
    "pred_TW": ("Taiwan (TW)",      "#43A047"),
    "pred_CN": ("China (CN)",       "#FF9800"),
    "pred_None": ("Open Sea",       "#78909C"),
}

# ── load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(CSV)
# only rows with predictions (test set ≥ 2020, excluding 2025 which has no preds)
test = df.dropna(subset=["pred_count"]).copy()
test["date"] = pd.to_datetime(test[["year","month"]].assign(day=1))
test["residual"] = test["pred_count"] - test["actual_count"]

years = sorted(test["year"].unique())

# ══════════════════════════════════════════════════════════════════════════
# CHART 1 — Monthly Actual vs Predicted (all test years, faceted)
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(len(years), 1, figsize=(14, 3.2 * len(years)),
                         sharex=True, sharey=True)
if len(years) == 1:
    axes = [axes]

for ax, yr in zip(axes, years):
    sub = test[test["year"] == yr].sort_values("month")
    x = np.arange(12)
    w = 0.35
    bars_a = ax.bar(x - w/2, sub["actual_count"], w, label="Thực tế (Actual)",
                    color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)
    bars_p = ax.bar(x + w/2, sub["pred_count"], w, label="Dự báo (Predicted)",
                    color=ORANGE, edgecolor="white", linewidth=0.5, zorder=3)

    # value labels
    for bar in bars_a:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.15, f"{int(h)}",
                    ha="center", va="bottom", fontsize=8, color=BLUE, fontweight="bold")
    for bar in bars_p:
        h = bar.get_height()
        if h > 0.3:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.15, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=8, color="#E65100", fontweight="bold")

    # highlight peak season (Jul–Nov)
    ax.axvspan(5.5, 10.5, alpha=0.06, color="red", zorder=0)

    actual_total = sub["actual_count"].sum()
    pred_total = sub["pred_count"].sum()
    mae = np.mean(np.abs(sub["residual"]))
    ax.set_ylabel("Số lượng bão")
    ax.set_title(f"{yr}  —  Actual={int(actual_total)}  Predicted={pred_total:.1f}  MAE={mae:.2f}",
                 fontsize=12, loc="left")
    ax.set_ylim(0, max(sub["actual_count"].max(), sub["pred_count"].max()) + 2)

axes[0].legend(loc="upper left", framealpha=0.9, fontsize=10)
axes[-1].set_xticks(range(12))
axes[-1].set_xticklabels(MONTH_LABELS)
axes[-1].set_xlabel("Tháng")

overall_mae = np.mean(np.abs(test["residual"]))
fig.suptitle(f"Chi tiết Dự báo hàng tháng — Test Set ≥ 2020  (Overall MAE = {overall_mae:.2f})",
             fontsize=16, fontweight="bold", y=1.01, color="#1565C0")
fig.tight_layout()
fig.savefig(DOCS / "chart_monthly_all_years.png", dpi=180, bbox_inches="tight",
            facecolor="white", pad_inches=0.3)
plt.close(fig)
print(f"✅ Saved chart_monthly_all_years.png")


# ══════════════════════════════════════════════════════════════════════════
# CHART 2 — Error Analysis (Residual distribution + scatter)
# ══════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# --- Residual histogram ---
residuals = test["residual"].values
ax1.hist(residuals, bins=np.arange(residuals.min()-0.5, residuals.max()+1.5, 0.5),
         color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.5, zorder=3)
ax1.axvline(0, color=RED, linewidth=1.5, linestyle="--", alpha=0.8, zorder=4)
ax1.axvline(residuals.mean(), color=ORANGE, linewidth=1.5, linestyle="-", alpha=0.8, zorder=4,
            label=f"Mean = {residuals.mean():+.2f}")
ax1.set_xlabel("Residual (Predicted − Actual)")
ax1.set_ylabel("Frequency")
ax1.set_title("Phân phối sai số dự báo", loc="left")
ax1.legend(fontsize=10)

# --- Scatter actual vs predicted ---
ax2.scatter(test["actual_count"], test["pred_count"], c=BLUE, alpha=0.6,
            edgecolors="white", linewidths=0.5, s=60, zorder=3)
lim = max(test["actual_count"].max(), test["pred_count"].max()) + 1
ax2.plot([0, lim], [0, lim], "--", color=RED, linewidth=1.5, alpha=0.7, label="Perfect prediction")
ax2.set_xlabel("Actual Storm Count")
ax2.set_ylabel("Predicted Storm Count")
ax2.set_title("Actual vs Predicted — Scatter", loc="left")
ax2.set_xlim(-0.5, lim)
ax2.set_ylim(-0.5, lim)
ax2.set_aspect("equal")
ax2.legend(fontsize=10)

# correlation
from scipy.stats import pearsonr
r, p = pearsonr(test["actual_count"], test["pred_count"])
ax2.text(0.05, 0.92, f"r = {r:.3f}  (p < {max(p,1e-10):.1e})",
         transform=ax2.transAxes, fontsize=11, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

fig.suptitle("Phân tích Sai số — Error Analysis", fontsize=15, fontweight="bold",
             y=1.02, color="#1565C0")
fig.tight_layout()
fig.savefig(DOCS / "chart_error_distribution.png", dpi=180, bbox_inches="tight",
            facecolor="white", pad_inches=0.3)
plt.close(fig)
print(f"✅ Saved chart_error_distribution.png")


# ══════════════════════════════════════════════════════════════════════════
# CHART 3 — Country-Level Landfall Predictions
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(len(years), 1, figsize=(14, 3.5 * len(years)),
                         sharex=True)
if len(years) == 1:
    axes = [axes]

for ax, yr in zip(axes, years):
    sub = test[test["year"] == yr].sort_values("month")
    x = np.arange(12)

    # stacked area
    bottom = np.zeros(12)
    for col, (label, color) in COUNTRY_COLS.items():
        vals = sub[col].values if col in sub.columns else np.zeros(12)
        ax.bar(x, vals, bottom=bottom, label=label, color=color, alpha=0.85,
               edgecolor="white", linewidth=0.3, width=0.72, zorder=3)
        bottom += vals

    # overlay actual count line
    ax.plot(x, sub["actual_count"].values, "o-", color="black", linewidth=2,
            markersize=5, label="Total Actual", zorder=5)

    ax.set_ylabel("Predicted Landfalls")
    ax.set_title(f"{yr}", fontsize=12, fontweight="bold", loc="left")
    ax.set_ylim(0, max(bottom.max(), sub["actual_count"].max()) + 1.5)

axes[0].legend(loc="upper left", fontsize=9, ncol=3, framealpha=0.9)
axes[-1].set_xticks(range(12))
axes[-1].set_xticklabels(MONTH_LABELS)
axes[-1].set_xlabel("Tháng")

fig.suptitle("Dự báo Đổ bộ theo Quốc gia — Country-Level Landfall Predictions",
             fontsize=15, fontweight="bold", y=1.01, color="#1565C0")
fig.tight_layout()
fig.savefig(DOCS / "chart_country_landfall.png", dpi=180, bbox_inches="tight",
            facecolor="white", pad_inches=0.3)
plt.close(fig)
print(f"✅ Saved chart_country_landfall.png")


# ══════════════════════════════════════════════════════════════════════════
# CHART 4 — Seasonal Pattern (Climatology)
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5.5))

# All years (1983–2024) actual climatology
all_actual = df.groupby("month")["actual_count"].mean()
# Test years climatology
test_actual = test.groupby("month")["actual_count"].mean()
test_pred   = test.groupby("month")["pred_count"].mean()

x = np.arange(1, 13)
ax.fill_between(x, 0, all_actual.reindex(x).values, alpha=0.12, color=BLUE, zorder=1,
                label="Climatology 1983–2024 (mean)")
ax.plot(x, all_actual.reindex(x).values, "-", color=BLUE, alpha=0.4, linewidth=1.5, zorder=2)

ax.plot(x, test_actual.reindex(x).values, "o-", color=BLUE, linewidth=2.5,
        markersize=8, label="Test Actual (2020–2024 mean)", zorder=4)
ax.plot(x, test_pred.reindex(x).values, "s--", color=ORANGE, linewidth=2.5,
        markersize=8, label="Test Predicted (2020–2024 mean)", zorder=4)

# highlight peak region
ax.axvspan(6.5, 11.5, alpha=0.06, color="red", zorder=0)
ax.text(9, ax.get_ylim()[1] * 0.92, "Đỉnh mùa bão", ha="center",
        fontsize=11, color="red", fontstyle="italic", alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(MONTH_LABELS)
ax.set_xlabel("Tháng")
ax.set_ylabel("Số lượng bão trung bình")
ax.set_title("Chu kỳ mùa bão — Seasonal Climatology vs Predictions",
             fontsize=14, fontweight="bold", color="#1565C0")
ax.legend(fontsize=10, framealpha=0.9)

fig.tight_layout()
fig.savefig(DOCS / "chart_seasonal_pattern.png", dpi=180, bbox_inches="tight",
            facecolor="white", pad_inches=0.3)
plt.close(fig)
print(f"✅ Saved chart_seasonal_pattern.png")

print(f"\n🎉 All charts saved to {DOCS}/")
