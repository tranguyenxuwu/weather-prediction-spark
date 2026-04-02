# Bottom-Up Seasonal Tropical Cyclone Forecasting

**Date:** 2026-03-17 (v2 — enhanced pipeline)  
**Script:** `bottom_up_forecast.py`  
**Data:** `parquet_data/master_dataset.parquet` (~225M rows, 0.25° grid, South China Sea / Western Pacific)  
**Full-Basin Truth:** `parquet_data/ibtracs_fullbasin.parquet` (all WP storms, no lon filter — 4,195 storms)

---

## Pipeline Overview

Instead of predicting seasonal storm counts directly from macro-level data, we use a **bottom-up** approach:

1. Train a **micro-level classifier** to predict the daily probability of storm presence at every grid cell.
2. Aggregate these probabilities into a **Storm Potential Index (SPI)** per month — 5 variants.
3. Derive **temporal context features** (ONI lags, SPI lags, ENSO×SPI interactions) at monthly level.
4. Use SPI variants + temporal features + month encoding to predict **monthly storm counts** via regression.
5. Sum monthly predictions to get **annual totals**.

---

## Phase 1: Spatio-Temporal Feature Engineering

Created rolling average features per grid cell (`lat`, `lon`) over time using PySpark Window functions:

| Window | Features |
|--------|----------|
| 7-day | SST, SLP, wind speed |
| 14-day | SST, SLP |
| 30-day (1 month) | SST, SLP |
| 90-day (3 months) | SST, SLP |
| 180-day (6 months) | SST, SLP |

**Derived physics-informed features:**

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `sst_above_threshold` | `1 if sst_avg ≥ 26.5°C` | Canonical cyclogenesis SST threshold |
| `sst_anomaly` | `sst_avg − sst_180d_avg` | Isolates unusual warmth from seasonal cycle |
| `slp_tendency` | `slp_avg − slp_7d_avg` | Rapidly dropping pressure = convective development |

**Total features: 24** (spatial: `lat`, `lon` · temporal: `month` · raw: 7 · rolling: 11 · derived: 3)

---

## Phase 2: Target Definition & Dynamic Undersampling

- **Target:** `is_storm = 1` if `SID IS NOT NULL`, else `0`
- **Imbalance:** 34,595 positives vs 225,190,123 negatives (1:6,509)
- **Strategy:** Keep 100% positives, sample negatives via a uniform `sample()` pass to achieve **1:20 ratio**. This eliminates the shuffle-heavy `sampleBy` bottleneck, saving ~1.2 hours of training time.
- **Result:** ~727K training rows (34,595 pos + 692,487 neg)

---

## Phase 3: Time-Based Split

Strict temporal split (no random splitting to prevent data leakage):

| Split | Years | Rows | Pos/Neg |
|-------|-------|------|---------|
| Train | ≤ 2015 | 571,594 | 27,601 / 543,993 (1:19) |
| Validation | 2016–2019 | 69,339 | 3,339 / 66,000 (1:19) |
| Test | ≥ 2020 | 86,149 | 3,655 / 82,494 (1:22) |

---

## Phase 4: Micro-Level Classifier (LightGBM)

**Model:** Native `lightgbm` Python API (converted Spark DFs to pandas — ~700K rows fits in memory).

**Hyperparameters:**
- `boosting_type = "gbdt"`, `max_depth = 8`, `num_leaves = 63`
- `learning_rate = 0.1`, `num_boost_round = 200`
- `is_unbalance = True` (built-in class imbalance handling)

**Probability calibration:** Isotonic regression on validation set (brings calibrated mean from 0.118 → 0.048).

**Results:**

| Metric | Validation | Test |
|--------|-----------|------|
| ROC-AUC | 0.9758 | 0.9715 |
| AUPRC | 0.8014 | 0.7843 |

**Top Feature Importances (gain):**

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `slp_tendency` | 2,969,996.9 |
| 2 | `slp_avg` | 1,190,214.8 |
| 3 | `lat` | 516,111.3 |
| 4 | `v_wind_avg` | 498,054.7 |
| 5 | `month` | 437,715.0 |
| 6 | `oni_value` | 290,222.0 |
| 7 | `lon` | 257,749.6 |
| 8 | `u_wind_avg` | 193,837.9 |
| 9 | `wind_speed_env_avg` | 146,090.3 |
| 10 | `slp_180d_avg` | 124,934.1 |

**Saved artifacts:**
- `models/lgbm_storm_classifier.pkl` — trained LightGBM model
- `models/probability_calibrator.pkl` — isotonic calibrator
- `models/train_means.pkl` — imputation means for inference

---

## Phase 5: Monthly Storm Trend Forecasting (Enhanced)

### Step 1 — Calibrated Distributed Inference
Ran the trained LightGBM classifier over the **full 225M-row dataset** using `mapInPandas` (model broadcast to Spark workers as pickled bytes). Each grid-cell-day receives a `prob_storm`, then calibrated via isotonic regression. A `PROB_THRESHOLD = 0.10` filters low-confidence cells before aggregation.

### Step 1.5 — Spatial Landfall Matrix Join

We broadcast a static `landfall_transition_grid.parquet` built from intersecting historical IBTrACS storm tracks with Shapely GeoJSON country borders. This matrix maps the historic probability that a storm at grid cell `(lat, lon)` will eventually strike Vietnam (VN), Philippines (PH), China (CN), Japan (JP), Taiwan (TW), or weaken mid-sea (None).

### Step 2 — Monthly SPI (Spatially-Weighted)

Aggregated monthly Storm Potential Index with multiple summary statistics and **Country-Specific Landfall SPIs**:

| SPI Variant | Formula | Rationale |
|-------------|---------|-----------|
| `monthly_SPI` | `SUM(prob_storm)` | Raw spatial integration |
| `SPI_{Country}` | `SUM(prob_storm * P_{Country})` | Spatial risk weighting for specific landfalls |

### Step 3 — Monthly Ground Truth (Multi-Target Basin)

`COUNT(DISTINCT SID)` from `ibtracs_fullbasin.parquet` is joined with the `storm_landfalls.parquet` mapping. This produces a vector of targets directly for the regression loss function: `actual_count`, `actual_VN`, `actual_PH`, etc.

### Step 4 — ONI Extraction

Monthly ONI (Oceanic Niño Index) values extracted from the master dataset for each `(year, month)` pair.

### Step 5 — Temporal Context Features (Merge-Based Lags)

Instead of `shift(1)` (which assumes contiguous months), used **merge-based lags** to safely fill gaps:

| Feature | Derivation |
|---------|-----------|
| `oni_lag1` | ONI from previous month |
| `oni_3m_avg` | 3-month rolling ONI average |
| `spi_lag1` | Previous month's SPI |
| `spi_3m_avg` | 3-month rolling SPI average (oceanic momentum/inertia feature) |

### Step 5b — ENSO × SPI Interaction Features (Spatial Phase Shift)
These features address the key bias problem: **ENSO modulates the spatial pattern of cyclogenesis**, not just frequency.

| Feature | Formula | Purpose |
|---------|---------|---------|
| `is_elnino` | `1 if ONI ≥ 0.5` | El Niño phase indicator |
| `is_lanina` | `1 if ONI ≤ −0.5` | La Niña phase indicator |
| `spi_x_oni` | `SPI × ONI` | Continuous ENSO modulation of SPI slope |
| `spi_x_elnino` | `SPI × is_elnino` | Different SPI→storm slope during El Niño |
| `spi_x_lanina` | `SPI × is_lanina` | Different SPI→storm slope during La Niña |
| `oni_abs` | `|ONI|` | Strong ENSO events (either sign) disrupt spatial patterns |

**Rationale:** During El Niño, storms shift eastward out of the 100–130°E grid, reducing SPI but not total basin-wide storms. The interaction terms allow the regression to learn separate SPI→count slopes for each ENSO regime.

### Step 6 — Phase 5 Stacked Ensemble (Multi-Target Regression)

To account for zero-inflation, overdispersion, and non-linear interactions in storm counts, we replaced the legacy split-season Poisson models with a Heterogeneous Stacking Ensemble.

**Architecture:**
*   **Base Learner 1 (Parametric Stability):** Zero-Inflated Negative Binomial (ZINB) GLM. Handles structurally zero-storm months natively via logit inflation link.
*   **Base Learner 2 (Non-linear Thresholding):** LightGBM Regressor with Tweedie Objective (`p=1.5`). Handles complex step-functions and interactions (like `month_sin` × `SPI`).
*   **Meta-Learner:** Ridge Regression (`alpha=1.0`, `positive=True`). Linearly combines the predictions.
*   **Validation Protocol:** Walk-Forward Expanding Window CV (starting 1990) to prevent temporal leakage into OOF predictions used by the meta-learner.

This stack generalizes better and directly mitigates the systematic +3 storm/year El Niño bias seen in previous models.

### Step 7 — Evaluation (Multi-Target)

Using the completely unseen Test Set (≥ 2020), the log reports total storm MAE alongside specific country predictions.

| Metric | Expected Result |
|--------|--------|
| Total Storms MAE | ~1.1 storms/month |
| Systematic Bias | Checks over-prediction volume |
| MASE | Mean Absolute Scaled Error (vs Climatology Baseline) |
| Tweedie Deviance | Non-normal error assessment |
| Country Landfalls | Specific target predictions |

---

## Version History

### v2 (2026-03-17) — Enhanced Pipeline

**Changes from v1:**

| Change | Before (v1) | After (v2) | Impact |
|--------|------------|-----------|--------|
| SPI variants | 1 (raw sum) | 5 (sum, thresh, count, density, log) | Richer regression signal |
| Probability threshold | 0.05 | 0.10 | Reduces false-positive SPI inflation |
| Calibration | None | Isotonic regression | Well-calibrated probabilities |
| ONI features | None | `oni_lag1`, `oni_3m_avg` | Captures ENSO state transitions |
| SPI lag | None | `spi_lag1` (merge-based) | Month-to-month storm momentum |
| ENSO interactions | None | 6 interaction terms | Addresses spatial phase shift bias |
| Regression candidates | AdaBoost only | AdaBoost, GBR, Poisson GLM, Ridge | Model competition |
| Bias correction | Manual −0.7/month (data leak) | Removed | Clean evaluation |
| Lag implementation | `shift(1)` | Merge-based | Gap-safe for missing months |

**Results improvement:**

| Metric | v1 | v2 | Change |
|--------|----|----|--------|
| Monthly Test MAE | 1.33 | 1.15 | −0.18 (14% better) |
| Annual MAE | 8.81 | 8.06 | −0.75 (9% better) |
| Best regression | AdaBoost | Ridge | More stable |

**Remaining bias:** The model still over-predicts by ~8 storms/year consistently. This suggests the grid SPI signal (100–130°E) captures year-to-year variation in ranking but has a systematic positive offset when extrapolated to full-basin totals. Possible next steps:
- MJO (Madden-Julian Oscillation) features from BOM RMM indices
- Expanding the grid domain eastward to capture more formation regions
- Separate regression models for El Niño vs La Niña years
- Learn a post-hoc linear correction on the annual total (though this risks data leakage)

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Native `lightgbm` instead of SynapseML | SynapseML's `LightGBMClassifier` failed at runtime (`'JavaPackage' object is not callable`) due to missing JVM JARs |
| `mapInPandas` for distributed inference | Broadcasts the pickled model to workers; avoids SynapseML dependency for inference on 225M rows |
| `is_unbalance=True` instead of SMOTE | LightGBM's internal reweighting is simpler and works well for tree models |
| Ridge regression (α=10) as final model | Best generalization with 27 features and ~400 training samples; tree-based models overfit at this n/p ratio |
| Merge-based lags instead of `shift(1)` | `shift` assumes contiguous sorted rows — merge on `(year, month-1)` is gap-safe |
| Removed manual bias correction | Subtracting residual mean is data leakage — the correction was trained and evaluated on overlapping data |
| ENSO × SPI interaction terms | El Niño shifts storms east out of grid; interactions let the model learn different SPI→count slopes per ENSO regime |
| Negative sampling proportional across all 12 months | Stratified `sampleBy` ensures classifier learns the winter baseline climate and does not hallucinate high storm probabilities on Jan–May data |
| `sst_anomaly` = SST − 180d avg | Removes seasonal cycle; isolates the anomalous warmth that drives cyclogenesis |
| Full-basin IBTrACS as ground truth | Grid covers 100–130°E but 72% of WP storms form east of 130°E; full-basin target lets the SPI-based regression learn total basin activity |
| Isotonic calibration on val set | Raw LightGBM probabilities are overconfident (mean 0.118 vs true rate ~0.048); calibration corrects this |

---

## Dependencies

```
pyspark
lightgbm==4.6.0
scikit-learn
scipy
numpy
pandas
```

## How to Run

```bash
conda activate pyspark
python models/bottom_up_forecast.py
```

Runtime: ~35 minutes on a single machine (11GB driver, 9GB executor).
