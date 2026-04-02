# WeatherPredict — Agent Instructions

## Project Summary

Bottom-up tropical cyclone forecasting pipeline. Predicts **monthly storm counts** by first classifying storm presence at every 0.25° grid cell per day, then rolling up probabilities into a Storm Potential Index (SPI).

**Domain:** South China Sea / Western Pacific, 1983–2024.

## Tech Stack

- **PySpark** — data engineering (rolling features, joins, distributed inference via `mapInPandas`)
- **LightGBM** (native Python API, not SynapseML) — micro-level storm classifier
- **scikit-learn** — monthly regression (Poisson GridSearchCV), probability calibration (isotonic), metrics
- **pandas / numpy / scipy** — local training, post-processing
- **Streamlit** — visualization dashboard (`app.py`, `app/app.py`)
- **Python 3.11+**, Conda env: `pyspark`

## Project Structure

```
WeatherPredict/
├── app.py                      # Streamlit dashboard — SST & Storms tabs
├── AGENTS.md                   # Agent instructions (this file)
├── requirements.txt            # Python dependencies
├── oni.csv                     # ONI index raw data
├── world-countries.json        # GeoJSON for map rendering
├── verification_map.png        # Output verification plot
│
├── app/
│   └── app.py                  # Phase 5 predictions dashboard (actual vs predicted)
│
├── models/                     # Modular pipeline + trained artifacts
│   ├── __init__.py             # Package init
│   ├── bottom_up_forecast.py   # CLI entry point (~180 lines)
│   ├── config.py               # Paths, Spark configs, feature lists, constants
│   ├── spark_session.py        # create_spark(), get_optimal_partitions()
│   ├── sanitize.py             # sanitize_inf_nan(), enforce_dtypes(), clamp_predictions()
│   ├── ensemble.py             # ZINB_Wrapper, Phase5_StackedEnsemble, expanding_window_cv()
│   ├── phase1_features.py      # Rolling feature engineering (persistent cache)
│   ├── phase2_sampling.py      # Undersampling (1:20 ratio)
│   ├── phase3_split.py         # Time-based split
│   ├── phase4_classifier.py    # LightGBM classifier + calibration
│   ├── phase5_rollup.py        # SPI inference, cache, multithreaded training, eval
│   ├── train.zsh               # Interactive training script
│   ├── lgbm_storm_classifier.pkl
│   ├── probability_calibrator.pkl
│   ├── phase5_ensemble.pkl
│   ├── phase5_monthly_cache.parquet  # Pre-calculated monthly data (~500 rows)
│   ├── ridge_monthly_model.pkl (legacy)
│   ├── monthly_predictions.csv
│   └── train_means.pkl
│
├── helpers/                    # Data engineering & utility scripts
│   ├── spatio_temporal_join.py # Builds master_dataset from raw sources
│   ├── data_convert.py         # CSV → Parquet conversion
│   ├── convert_sst.py          # NOAA SST NetCDF → Parquet
│   ├── preprocess_era5.py      # ERA5 GRIB → Parquet
│   ├── preprocess_ibtracs.py   # IBTrACS CSV → Parquet
│   ├── extract_landfall_grid.py # Generates map transition grid
│   ├── read_parquet.py         # Parquet inspection utility
│   ├── count_parquet_rows.py   # Row-count utility
│   ├── verify_visualization.py # Visualization sanity check
│   └── visualization_helper.py # Shared plotting helpers
│
├── cluster/                    # Spark Standalone cluster scripts
│   ├── start_master.sh         # Start Spark Master on this machine
│   ├── start_worker.sh         # Start Spark Worker (run on 2nd machine)
│   ├── stop_cluster.sh         # Stop all Spark daemons
│   └── sync_data.sh            # rsync data to worker machine
│
├── docs/                       # Documentation & reference
│   ├── train.md                # Training log and results
│   ├── cluster_setup.md        # 2-node cluster setup guide
│   ├── master_dataset_ml_guide.md  # Data dictionary & ML recommendations
│   └── master_plan.md          # Original project plan
│
├── parquet_data/               # master_dataset.parquet (~225M rows)
│   └── features_checkpoint.parquet  # Persistent Phase 1 feature cache
└── SPARK__DATA/                # Raw source data (ERA5, SST, IBTrACS, ONI)
```

## Key Files

| File | Purpose |
|------|---------|
| `models/bottom_up_forecast.py` | **CLI entry point.** Orchestrates 5 phases, supports `--prepare`, `--phase5`, `--legacy`, `--rebuild-features`. |
| `models/config.py` | **Single source of truth.** All paths, Spark configs, feature lists, constants. |
| `models/phase5_rollup.py` | **Phase 5 logic.** Pre-calculated cache, multithreaded 7-target training, evaluation. |
| `models/ensemble.py` | Stacked ensemble: ZINB + Tweedie LightGBM + Ridge meta-learner. |
| `helpers/spatio_temporal_join.py` | Joins ERA5 + NOAA SST + ONI + IBTrACS into `master_dataset.parquet`. Run once. |
| `docs/master_dataset_ml_guide.md` | Complete data dictionary and ML recommendations. **Read this first.** |
| `docs/train.md` | Documents pipeline decisions, hyperparams, and results. |
| `docs/cluster_setup.md` | 2-node Spark Standalone cluster setup guide. |

## Pipeline Phases

1. **Feature Engineering** (`phase1_features.py`) — rolling averages (7/14/30/90/180 day) + derived features. **Persistently cached** — computed once (~1.5h), loaded instantly (~30s) on rerun.
2. **Undersampling** (`phase2_sampling.py`) — 1:20 pos/neg via fast uniform `sample()`.
3. **Temporal Split** (`phase3_split.py`) — train ≤2015, val 2016–2019, test ≥2020. No random splits.
4. **LightGBM Classifier** (`phase4_classifier.py`) — trains on pandas (converted from Spark). Outputs `prob_storm`. ROC-AUC ~0.99.
5. **Monthly Roll-Up** (`phase5_rollup.py`) — `mapInPandas` inference + Landfall Spatial Transition Matrix → Country-specific SPIs → Phase 5 Stacked Ensemble (ZINB + Tweedie LightGBM + Ridge) → Country-level landfall predictions. **7 targets train in parallel** via ThreadPoolExecutor.

## Important Constraints

- **No SynapseML.** Use native `lightgbm` Python API only. SynapseML causes `JavaPackage` errors.
- **No random splits.** Time-based only — future data must never leak into training.
- **Inference uses `mapInPandas`** with a broadcasted pickled model. Don't try `spark.ml` transformers.
- **Data is huge (~225M rows).** Feature engineering happens in Spark; training happens in pandas after undersampling reduces to ~700K rows.
- **Predictions must be clipped at 0** — no negative storm counts.
- **Two-level caching:** Phase 1 features → `features_checkpoint.parquet` (persistent). Monthly data → `phase5_monthly_cache.parquet` (persistent). Both skip Spark on rerun.

## Running

```bash
conda activate pyspark

# Full pipeline (first run: ~45 min; cached: ~20 min)
python -m models.bottom_up_forecast

# Prepare monthly cache only — no training (~20 min first, ~5 min cached)
python -m models.bottom_up_forecast --prepare

# Phase 5 from cache — no Spark needed (~3 min)
python -m models.bottom_up_forecast --phase5

# Legacy Poisson mode from cache
python -m models.bottom_up_forecast --phase5 --legacy

# Force recompute Phase 1 rolling features
python -m models.bottom_up_forecast --rebuild-features

# Dashboards
streamlit run app.py                           # Data exploration (SST & Storms)
streamlit run app/app.py                       # Phase 5 predictions
```

### Using train.zsh

```bash
./models/train.zsh              # Interactive menu
./models/train.zsh bottom_up    # Full pipeline
./models/train.zsh prepare      # Build monthly cache
./models/train.zsh phase5       # Train from cache (~3 min)
./models/train.zsh bottom_up --rebuild-features  # Force recompute features
```

### Cluster Mode (2-node)

```bash
# Option 1: Automated (starts master, local worker, trains, stops)
./models/train.zsh cluster

# Option 2: Manual
./cluster/start_master.sh                         # On master machine
./cluster/start_worker.sh <MASTER_IP>              # On worker machine
SPARK_CLUSTER_MODE=cluster SPARK_MASTER_IP=<IP> python -m models.bottom_up_forecast
./cluster/stop_cluster.sh                          # When done
```

See `docs/cluster_setup.md` for full setup instructions.

## Common Pitfalls

- Duplicate columns in `mapInPandas`: deduplicate `inference_cols` with `list(dict.fromkeys(...))`.
- NaN imputation: use `models/train_means.pkl` (means from training set only).
- The `month` column is both a feature and a group-by key — handle carefully in Phase 5.
- **Redundant Checkpointing**: Phase 1 already breaks the PySpark lineage DAG by writing to `features_checkpoint.parquet`. Do not checkpoint `full_df` again in Phase 5 before `mapInPandas`. Use native `.na.fill()` and let `mapInPandas` stream directly.
- **ZINB numerical stability**: statsmodels ZINB can overflow to inf. All predictions are clamped via `sanitize.clamp_predictions()`.
- **Thread safety**: Phase 5 model training uses ThreadPoolExecutor. Each thread creates independent model instances — no shared mutable state. Prediction assignment happens in main thread only.
