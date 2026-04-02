"""
Bottom-Up Seasonal Tropical Cyclone Forecasting Pipeline
=========================================================
5-phase ML pipeline:
  Phase 1: Spatio-Temporal Feature Engineering (rolling averages)
  Phase 2: Target Definition & Stratified Undersampling
  Phase 3: Time-Based Split
  Phase 4: Micro-Level Classifier (LightGBM → prob_storm)
  Phase 5: Monthly Storm Trend Forecasting (SPI → monthly count)

Usage:
    conda activate pyspark
    python -m models.bottom_up_forecast                     # Full pipeline (Phases 1–5)
    python -m models.bottom_up_forecast --prepare           # Phases 1–2 + inference cache only
    python -m models.bottom_up_forecast --phase5            # Phase 5 from cache (fast, no Spark)
    python -m models.bottom_up_forecast --phase5 --legacy   # Legacy Poisson mode from cache
"""

import argparse
import gc
import logging
import pickle
import shutil
import sys
import time
from pathlib import Path

from .config import MODEL_DIR, MODEL_PATH, SPARK_LOCAL_DIR
from .spark_session import create_spark
from .phase1_features import phase1_feature_engineering
from .phase2_sampling import phase2_undersampling
from .phase3_split import phase3_time_split
from .phase4_classifier import phase4_classifier
from .phase5_rollup import (
    prepare_monthly_cache, load_monthly_cache,
    derive_features, train_models, evaluate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("BottomUpForecast")


def _load_phase4_artifacts():
    """Load Phase 4 model artifacts from disk."""
    for artifact_name, artifact_path in [
        ("LightGBM model", MODEL_PATH),
        ("Calibrator", str(MODEL_DIR / "probability_calibrator.pkl")),
        ("Train means", str(MODEL_DIR / "train_means.pkl")),
    ]:
        if not Path(artifact_path).exists():
            log.error(f"   ✗ {artifact_name} not found: {artifact_path}")
            log.error("   Run the full pipeline first.")
            sys.exit(1)

    with open(MODEL_PATH, "rb") as f:
        lgbm_model = pickle.load(f)
    with open(str(MODEL_DIR / "probability_calibrator.pkl"), "rb") as f:
        calibrator = pickle.load(f)
    with open(str(MODEL_DIR / "train_means.pkl"), "rb") as f:
        train_means = pickle.load(f)

    log.info("   ✅ Loaded model, calibrator, and train means from disk.")
    return lgbm_model, calibrator, train_means


def _detect_incomplete_years(monthly_data):
    """Find years with < 6 months of SPI data (unreliable for eval)."""
    months_per_year = monthly_data.groupby("year")["month"].count()
    incomplete = set(months_per_year[months_per_year < 6].index)
    if incomplete:
        log.warning(f"   ⚠️  Incomplete data years (< 6 months): "
                    f"{sorted(incomplete)} — excluded from test evaluation")
    return incomplete


def main():
    parser = argparse.ArgumentParser(
        description="Bottom-Up Tropical Cyclone Forecasting Pipeline"
    )
    parser.add_argument(
        "--phase5", action="store_true",
        help="Phase 5 only from cache (no Spark needed, ~3 min)",
    )
    parser.add_argument(
        "--prepare", action="store_true",
        help="Run Phases 1-2 + inference, save monthly cache (no training)",
    )
    parser.add_argument(
        "--legacy", action="store_true",
        help="Use legacy split-season Poisson instead of stacked ensemble",
    )
    parser.add_argument(
        "--rebuild-features", action="store_true",
        help="Force recompute Phase 1 rolling features (ignores cache)",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Bottom-Up Seasonal Tropical Cyclone Forecasting")
    log.info("=" * 60)

    t_total = time.time()

    try:
        if args.phase5:
            # ═══════════════════════════════════════════════════════
            # FAST MODE: Load cache → train → evaluate (no Spark)
            # ═══════════════════════════════════════════════════════
            log.info("   🚀 Phase 5 fast mode — loading pre-calculated cache...")
            monthly_data = load_monthly_cache()
            incomplete_years = _detect_incomplete_years(monthly_data)

            monthly_data, feature_columns = derive_features(
                monthly_data, legacy_mode=args.legacy
            )
            target_results, models_dict, monthly_data, train_mask, test_mask = (
                train_models(
                    monthly_data, feature_columns, incomplete_years,
                    legacy_mode=args.legacy, parallel=True,
                )
            )
            test_mae, annual_summary, annual_mae = evaluate(
                monthly_data, target_results, models_dict,
                train_mask, test_mask, legacy_mode=args.legacy,
            )

        elif args.prepare:
            # ═══════════════════════════════════════════════════════
            # PREPARE MODE: Spark inference → save cache (no training)
            # ═══════════════════════════════════════════════════════
            Path(SPARK_LOCAL_DIR).mkdir(parents=True, exist_ok=True)
            spark = create_spark()

            try:
                log.info("   📦 Prepare mode — building monthly cache...")
                lgbm_model, calibrator, train_means = _load_phase4_artifacts()

                full_df = phase1_feature_engineering(spark, rebuild=args.rebuild_features)
                gc.collect()

                _, full_df = phase2_undersampling(full_df)
                spark.catalog.clearCache()
                gc.collect()

                prepare_monthly_cache(
                    spark, lgbm_model, calibrator, train_means, full_df
                )
                log.info("\n" + "=" * 60)
                log.info("✅ CACHE PREPARED — run with --phase5 for fast training")
                log.info("=" * 60)
            finally:
                spark.stop()
                _cleanup_spark_tmp()

        else:
            # ═══════════════════════════════════════════════════════
            # FULL PIPELINE: Phases 1–5
            # ═══════════════════════════════════════════════════════
            Path(SPARK_LOCAL_DIR).mkdir(parents=True, exist_ok=True)
            spark = create_spark()

            try:
                # ── Phase 1 ──
                full_df = phase1_feature_engineering(spark, rebuild=args.rebuild_features)
                gc.collect()

                # ── Phase 2 ──
                training_base, full_df = phase2_undersampling(full_df)
                training_base.cache()
                log.info("   Cached training base.\n")
                gc.collect()

                # ── Phase 3 ──
                train_df, val_df, test_df = phase3_time_split(training_base)
                train_df.cache()
                val_df.cache()
                test_df.cache()
                gc.collect()

                # ── Phase 4 ──
                lgbm_model, calibrator, train_means, metrics = (
                    phase4_classifier(spark, train_df, val_df, test_df)
                )

                # Free caches before full inference
                train_df.unpersist()
                val_df.unpersist()
                test_df.unpersist()
                training_base.unpersist()
                spark.catalog.clearCache()
                gc.collect()
                log.info("   Cleared all Spark caches before Phase 5.")

                # ── Phase 5: prepare cache + train ──
                monthly_data = prepare_monthly_cache(
                    spark, lgbm_model, calibrator, train_means, full_df
                )
            finally:
                spark.stop()
                _cleanup_spark_tmp()

            # Training and evaluation (no Spark needed)
            incomplete_years = _detect_incomplete_years(monthly_data)
            monthly_data, feature_columns = derive_features(
                monthly_data, legacy_mode=args.legacy
            )
            target_results, models_dict, monthly_data, train_mask, test_mask = (
                train_models(
                    monthly_data, feature_columns, incomplete_years,
                    legacy_mode=args.legacy, parallel=True,
                )
            )
            test_mae, annual_summary, annual_mae = evaluate(
                monthly_data, target_results, models_dict,
                train_mask, test_mask, legacy_mode=args.legacy,
            )

        # ── Final Summary ──
        if not args.prepare:
            log.info("\n" + "=" * 60)
            log.info("✅ PIPELINE COMPLETE")
            log.info("=" * 60)
            log.info(f"   LightGBM model   : {MODEL_PATH}")
            log.info(f"   Monthly Test MAE : {test_mae:.2f} storms/month")
            log.info(f"   Annual Test MAE  : {annual_mae:.2f} storms/year")
            log.info(f"   Total time       : {time.time() - t_total:.0f}s")

    except Exception as e:
        log.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


def _cleanup_spark_tmp():
    """Remove Spark temp directory."""
    tmp = Path(SPARK_LOCAL_DIR)
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
