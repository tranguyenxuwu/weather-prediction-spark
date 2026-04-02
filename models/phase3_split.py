"""
Phase 3: Time-Based Split.
"""

import logging
from collections import defaultdict

from pyspark.sql import functions as F

log = logging.getLogger("BottomUpForecast")


def phase3_time_split(training_base):
    """
    Strict temporal split — NO random splitting.
    Train: year <= 2015 | Val: 2016–2019 | Test: year >= 2020

    Returns:
        train_df, val_df, test_df: Spark DataFrames
    """
    log.info("=" * 60)
    log.info("PHASE 3: Time-Based Split")
    log.info("=" * 60)

    train_df = training_base.filter(F.col("year") <= 2015)
    val_df   = training_base.filter(F.col("year").between(2016, 2019))
    test_df  = training_base.filter(F.col("year") >= 2020)

    # Single-pass: all split × class counts at once
    split_stats = (
        training_base
        .withColumn("_split",
            F.when(F.col("year") <= 2015, F.lit("Train"))
             .when(F.col("year").between(2016, 2019), F.lit("Val"))
             .otherwise(F.lit("Test"))
        )
        .groupBy("_split", "is_storm")
        .count()
        .collect()
    )
    stats = defaultdict(lambda: {"pos": 0, "neg": 0, "total": 0})
    for row in split_stats:
        key = "pos" if row["is_storm"] == 1 else "neg"
        stats[row["_split"]][key] = row["count"]
        stats[row["_split"]]["total"] += row["count"]

    for name in ["Train", "Val", "Test"]:
        s = stats[name]
        ratio = s["neg"] // max(s["pos"], 1)
        log.info(f"   {name}: {s['total']:,} rows — "
                 f"{s['pos']:,} pos / {s['neg']:,} neg (1:{ratio})")

    return train_df, val_df, test_df
