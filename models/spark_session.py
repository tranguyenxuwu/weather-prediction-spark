"""
Spark session creation and cluster utilities.
"""

import logging

from pyspark.sql import SparkSession

from .config import (
    CLUSTER_MODE, MASTER_IP,
    SPARK_CONFIG_LOCAL, SPARK_CONFIG_CLUSTER,
)

log = logging.getLogger("BottomUpForecast")


def create_spark():
    """Create SparkSession — auto-selects local or cluster config."""
    config = SPARK_CONFIG_CLUSTER if CLUSTER_MODE == "cluster" else SPARK_CONFIG_LOCAL
    mode_label = (f"cluster (spark://{MASTER_IP}:7077)"
                  if CLUSTER_MODE == "cluster" else "local[*]")
    log.info(f"   Spark mode: {mode_label}")

    builder = SparkSession.builder.appName("BottomUpForecast")
    for k, v in config.items():
        builder = builder.config(k, v)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    if CLUSTER_MODE == "cluster":
        sc = spark.sparkContext
        log.info(f"   Cluster cores: {sc.defaultParallelism}")
        try:
            log.info(f"   Executors: {len(sc._jsc.sc().getExecutorMemoryStatus())}")
        except (AttributeError, Exception) as e:
            log.warning(f"   Could not query executor status: {e}")
    return spark


def get_optimal_partitions(spark):
    """Calculate optimal partition count from cluster resources.
    Rule of thumb: 3 partitions per available core.
    Minimum 64 to avoid too-large partitions on small clusters.
    """
    total_cores = spark.sparkContext.defaultParallelism
    return max(64, total_cores * 3)
