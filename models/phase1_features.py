"""
Phase 1: Spatio-Temporal Feature Engineering.
Creates rolling average features for each grid cell over time.

Features are saved to a persistent parquet cache so they only need
to be computed once. Subsequent runs skip directly to the cached output.
"""

import logging
import time
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType

from .config import INPUT_PATH, SPARK_LOCAL_DIR, ALL_FEATURES, FEATURES_CACHE_PATH

log = logging.getLogger("BottomUpForecast")


def phase1_feature_engineering(spark, rebuild=False):
    """
    Create rolling average features for each grid cell over time.
    Windows: 7d, 14d, 30d (1mo), 90d (3mo), 180d (6mo).

    If a persistent cache exists at FEATURES_CACHE_PATH and rebuild=False,
    skips all computation and loads directly (~30s vs ~1.5h).

    Args:
        spark: SparkSession
        rebuild: if True, recompute features even if cache exists

    Returns a Spark DataFrame with all features.
    """
    log.info("=" * 60)
    log.info("PHASE 1: Spatio-Temporal Feature Engineering")
    log.info("=" * 60)

    # ── Check persistent cache ──
    if not rebuild and Path(FEATURES_CACHE_PATH).exists():
        log.info(f"   ✅ Persistent feature cache found — loading directly")
        log.info(f"      {FEATURES_CACHE_PATH}")
        t0 = time.time()
        df = spark.read.parquet(FEATURES_CACHE_PATH).repartition(200)
        elapsed = time.time() - t0
        log.info(f"   Loaded {len(df.columns)} columns in {elapsed:.0f}s "
                 f"(skipped ~1.5h of rolling window computation)")
        return df

    t0 = time.time()
    log.info("   No persistent cache — computing rolling features from scratch...")

    # Column pruning: only read the columns we actually need (~40% less I/O)
    NEEDED_COLS = [
        "lat", "lon", "date", "year", "month", "SID",
        "sst_avg", "slp_avg", "u_wind_avg", "v_wind_avg",
        "wind_speed_env_avg", "oni_value", "enso_phase",
    ]
    df = spark.read.parquet(INPUT_PATH).select(*NEEDED_COLS)
    log.info(f"   Loaded master dataset (pruned to {len(NEEDED_COLS)} columns)")

    # Cast date to timestamp for Window ordering (seconds since epoch)
    df = df.withColumn("date_ts", F.col("date").cast("timestamp").cast("long"))

    # Window partitioned by grid cell, ordered by date
    rolling_configs = {
        7:   [("sst_avg", "sst_7d_avg"), ("slp_avg", "slp_7d_avg"),
              ("wind_speed_env_avg", "wind_env_7d_avg")],
        14:  [("sst_avg", "sst_14d_avg"), ("slp_avg", "slp_14d_avg")],
        30:  [("sst_avg", "sst_30d_avg"), ("slp_avg", "slp_30d_avg")],
        90:  [("sst_avg", "sst_90d_avg"), ("slp_avg", "slp_90d_avg")],
        180: [("sst_avg", "sst_180d_avg"), ("slp_avg", "slp_180d_avg")],
    }

    for window_days, features in rolling_configs.items():
        w = (Window.partitionBy("lat", "lon")
             .orderBy("date_ts")
             .rowsBetween(-(window_days - 1), 0))
        for src_col, new_col in features:
            df = df.withColumn(new_col, F.avg(F.col(src_col)).over(w))
        log.info(f"   ✅ {window_days}-day rolling averages computed")

    # ── Derived physics-informed features ──
    df = df.withColumn(
        "sst_above_threshold",
        F.when(F.col("sst_avg") >= 26.5, F.lit(1))
        .otherwise(F.lit(0)).cast(IntegerType())
    )
    df = df.withColumn(
        "sst_anomaly",
        F.col("sst_avg") - F.col("sst_180d_avg")
    )
    df = df.withColumn(
        "slp_tendency",
        F.col("slp_avg") - F.col("slp_7d_avg")
    )
    log.info("   ✅ Derived features computed (sst_above_threshold, "
             "sst_anomaly, slp_tendency)")

    # Drop the helper timestamp column
    df = df.drop("date_ts")

    # ── SAVE TO PERSISTENT CACHE ──
    # This replaces the old .spark_tmp checkpoint that was deleted every run.
    # Writing once saves ~1.5h on every subsequent run.
    log.info(f"   Saving persistent feature cache...")
    log.info(f"      → {FEATURES_CACHE_PATH}")
    df.write.mode("overwrite").parquet(FEATURES_CACHE_PATH)
    df = spark.read.parquet(FEATURES_CACHE_PATH).repartition(200)
    log.info("   ✅ Saved — lineage broken, clean DAG from here.")

    elapsed = time.time() - t0
    log.info(f"   Phase 1 complete in {elapsed:.0f}s")
    log.info(f"   Total features: {len(ALL_FEATURES)}")
    log.info(f"   Schema: {df.columns}")

    # Sample preview
    log.info("   Sample of engineered features:")
    df.filter(F.col("sst_avg").isNotNull()).select(
        "lat", "lon", "date", "month", "sst_avg",
        "sst_anomaly", "sst_above_threshold", "slp_tendency",
    ).show(5, truncate=False)

    return df
