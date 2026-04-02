"""
Centralized configuration for the Bottom-Up Tropical Cyclone Forecasting Pipeline.
All paths, Spark configs, feature lists, and constants live here.
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).resolve().parent.parent   # WeatherPredict/
PARQUET_DIR = BASE_DIR / "parquet_data"
INPUT_PATH  = str(PARQUET_DIR / "master_dataset.parquet")
MODEL_DIR   = BASE_DIR / "models"
MODEL_PATH  = str(MODEL_DIR / "lgbm_storm_classifier.pkl")

SPARK_LOCAL_DIR = str(BASE_DIR / ".spark_tmp")

# Persistent Phase 1 feature cache (avoids recomputing rolling features on 225M rows)
FEATURES_CACHE_PATH = str(PARQUET_DIR / "features_checkpoint.parquet")

# Pre-calculated monthly data cache (Phase 5 speedup)
CACHE_PATH = str(MODEL_DIR / "phase5_monthly_cache.parquet")

# Data files (absolute paths — fixes the relative-path CWD bug)
LANDFALL_TRANSITION_PATH = str(PARQUET_DIR / "landfall_transition_grid.parquet")
STORM_LANDFALLS_PATH     = str(PARQUET_DIR / "storm_landfalls.parquet")
FULLBASIN_PATH           = str(PARQUET_DIR / "ibtracs_fullbasin.parquet")

# ── Cluster mode ──────────────────────────────────────────────────────────────

CLUSTER_MODE = os.environ.get("SPARK_CLUSTER_MODE", "local")
MASTER_IP    = os.environ.get("SPARK_MASTER_IP", "localhost")

# ── Spark configs ─────────────────────────────────────────────────────────────

SPARK_CONFIG_LOCAL = {
    "spark.driver.memory": "10g",
    "spark.executor.memory": "8g",
    "spark.executor.memoryOverhead": "2g",
    "spark.sql.shuffle.partitions": "200",
    "spark.sql.parquet.compression.codec": "snappy",
    "spark.driver.maxResultSize": "1g",
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.execution.arrow.pyspark.enabled": "true",
    "spark.memory.fraction": "0.6",
    "spark.memory.storageFraction": "0.2",
    "spark.local.dir": SPARK_LOCAL_DIR,
    "spark.ui.showConsoleProgress": "true",
    "spark.sql.windowExec.buffer.in.memory.threshold": "2048",
    "spark.shuffle.spill.compress": "true",
    "spark.driver.extraJavaOptions":
        "-XX:+UseG1GC -XX:G1HeapRegionSize=16m "
        "-XX:InitiatingHeapOccupancyPercent=35 "
        "-XX:+ParallelRefProcEnabled "
        "-XX:+ExplicitGCInvokesConcurrent",
    "spark.executor.extraJavaOptions":
        "-XX:+UseG1GC -XX:G1HeapRegionSize=16m "
        "-XX:InitiatingHeapOccupancyPercent=35 "
        "-XX:+ParallelRefProcEnabled",
    "spark.sql.files.maxPartitionBytes": "64m",
}

SPARK_CONFIG_CLUSTER = {
    "spark.master": f"spark://{MASTER_IP}:7077",
    "spark.driver.memory": "9g",
    "spark.executor.memory": "9g",
    "spark.executor.memoryOverhead": "3g",
    "spark.executor.cores": "5",
    "spark.sql.shuffle.partitions": "128",
    "spark.sql.parquet.compression.codec": "snappy",
    "spark.driver.maxResultSize": "3g",
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.execution.arrow.pyspark.enabled": "true",
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
    "spark.default.parallelism": "15",
    "spark.memory.fraction": "0.6",
    "spark.memory.storageFraction": "0.2",
    "spark.local.dir": SPARK_LOCAL_DIR,
    "spark.ui.showConsoleProgress": "true",
    "spark.python.worker.reuse": "true",
    "spark.files.useFetchCache": "true",
}

# ── Feature lists ─────────────────────────────────────────────────────────────

SPATIAL_FEATURES = ["lat", "lon"]

TEMPORAL_FEATURES = ["month"]

RAW_FEATURES = [
    "sst_avg", "slp_avg", "u_wind_avg", "v_wind_avg",
    "wind_speed_env_avg", "oni_value", "enso_phase",
]

ROLLING_FEATURES = [
    "sst_7d_avg", "sst_14d_avg", "sst_30d_avg", "sst_90d_avg", "sst_180d_avg",
    "slp_7d_avg", "slp_14d_avg", "slp_30d_avg", "slp_90d_avg", "slp_180d_avg",
    "wind_env_7d_avg",
]

DERIVED_FEATURES = [
    "sst_above_threshold",   # SST ≥ 26.5°C (cyclogenesis threshold)
    "sst_anomaly",           # SST deviation from 180-day trailing mean
    "slp_tendency",          # 7-day pressure change rate
]

ALL_FEATURES = (
    SPATIAL_FEATURES + TEMPORAL_FEATURES +
    RAW_FEATURES + ROLLING_FEATURES + DERIVED_FEATURES
)

# ── Constants ─────────────────────────────────────────────────────────────────

MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

PROB_THRESHOLD = 0.10

LANDFALL_TARGETS = ["VN", "PH", "CN", "JP", "TW", "None"]

ALL_TARGETS = ["count"] + LANDFALL_TARGETS

# Prediction clamp — no single month has ever had >10 named storms;
# anything above 200 is clearly numerical overflow from ZINB.
PRED_CLAMP_UPPER = 200
