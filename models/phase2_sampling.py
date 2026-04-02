"""
Phase 2: Target Definition & Dynamic Undersampling.
"""

import logging
import time

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

log = logging.getLogger("BottomUpForecast")


def phase2_undersampling(df):
    """
    Create binary target and undersample negatives to ~1:20 ratio.
    Negatives are sampled uniformly — monthly fraction of empty grid cells
    is strictly proportional to days per month anyway.

    Returns:
        training_base: undersampled Spark DF for Phase 3/4
        full_df: full DF with is_storm column for Phase 5 inference
    """
    log.info("=" * 60)
    log.info("PHASE 2: Target Definition & Stratified Undersampling")
    log.info("=" * 60)

    t0 = time.time()

    # Binary target
    df = df.withColumn(
        "is_storm",
        F.when(F.col("SID").isNotNull(), F.lit(1))
        .otherwise(F.lit(0)).cast(IntegerType())
    )

    # Split positives and negatives
    positives = df.filter(F.col("is_storm") == 1)
    negatives = df.filter(F.col("is_storm") == 0)

    # Single-pass count
    label_counts = {
        row["is_storm"]: row["count"]
        for row in df.groupBy("is_storm").count().collect()
    }
    pos_count = label_counts.get(1, 0)
    neg_count = label_counts.get(0, 0)
    log.info(f"   Full dataset: {pos_count:,} positives, {neg_count:,} negatives")
    log.info(f"   Imbalance ratio: 1:{neg_count // max(pos_count, 1)}")

    # Uniform negative sampling
    target_neg = pos_count * 20
    fraction = target_neg / neg_count if neg_count > 0 else 0.001158
    log.info(f"   Target 1:20 → need {target_neg:,} negatives (frac: {fraction:.6f})")

    negatives_sampled = negatives.sample(
        withReplacement=False, fraction=fraction, seed=42
    )

    # Union
    training_base = positives.unionByName(negatives_sampled)

    # Estimate counts from sampling fraction
    final_pos = pos_count
    final_neg = int(neg_count * fraction)
    log.info(f"   ✅ Training base: ~{final_pos:,} positives, ~{final_neg:,} negatives")
    log.info(f"      Effective ratio: ~1:{final_neg // max(final_pos, 1)}")

    elapsed = time.time() - t0
    log.info(f"   Phase 2 complete in {elapsed:.0f}s")

    return training_base, df
