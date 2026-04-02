"""
Phase 4: Micro-Level Probability Classifier (LightGBM).
"""

import logging
import pickle
import time

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression

from .config import ALL_FEATURES, MODEL_DIR, MODEL_PATH

log = logging.getLogger("BottomUpForecast")


def phase4_classifier(spark, train_df, val_df, test_df):
    """
    Train native LightGBM (via pandas conversion) to predict grid-level
    storm probability. The undersampled data (~700K rows) fits in memory.

    Returns:
        model: trained LightGBM Booster
        calibrator: IsotonicRegression calibrator
        train_means: pd.Series of training set means for imputation
        results: dict with evaluation metrics
    """
    log.info("=" * 60)
    log.info("PHASE 4: Micro-Level Probability Classifier (LightGBM)")
    log.info("=" * 60)

    t0 = time.time()

    # ── Step 1: Convert Spark → Pandas ──
    log.info("   Step 1: Converting Spark DataFrames to Pandas...")
    cols_needed = ALL_FEATURES + ["is_storm"]

    train_pd = train_df.select(*cols_needed).toPandas()
    val_pd   = val_df.select(*cols_needed).toPandas()
    test_pd  = test_df.select(*cols_needed).toPandas()

    log.info(f"   Train: {len(train_pd):,}, Val: {len(val_pd):,}, "
             f"Test: {len(test_pd):,}")

    # ── Step 2: Handle NaNs (fill with column means from train) ──
    log.info("   Step 2: Imputing missing values (train-set means)...")
    train_means = train_pd[ALL_FEATURES].mean()
    train_pd[ALL_FEATURES] = train_pd[ALL_FEATURES].fillna(train_means)
    val_pd[ALL_FEATURES]   = val_pd[ALL_FEATURES].fillna(train_means)
    test_pd[ALL_FEATURES]  = test_pd[ALL_FEATURES].fillna(train_means)

    log.info(f"   Remaining NaNs in train: "
             f"{train_pd[ALL_FEATURES].isna().sum().sum()}")

    # ── Step 3: Create LightGBM datasets ──
    X_train = train_pd[ALL_FEATURES]
    y_train = train_pd["is_storm"]
    X_val   = val_pd[ALL_FEATURES]
    y_val   = val_pd["is_storm"]
    X_test  = test_pd[ALL_FEATURES]
    y_test  = test_pd["is_storm"]

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val   = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    # ── Step 4: Train LightGBM ──
    params = {
        "objective": "binary",
        "metric": ["auc", "average_precision"],
        "boosting_type": "gbdt",
        "max_depth": 8,
        "num_leaves": 63,
        "learning_rate": 0.1,
        "is_unbalance": True,
        "verbose": -1,
        "seed": 42,
        "n_jobs": -1,
    }

    log.info("   Step 3: Training LightGBM...")
    log.info(f"   Params: max_depth={params['max_depth']}, "
             f"num_leaves={params['num_leaves']}, is_unbalance=True")

    callbacks = [
        lgb.log_evaluation(period=20),
        lgb.early_stopping(stopping_rounds=30, verbose=True),
    ]

    model = lgb.train(
        params, lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    log.info(f"   ✅ Model trained! Best iteration: {model.best_iteration}")

    # ── Step 5: Save model ──
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    log.info(f"   Model saved to: {MODEL_PATH}")

    # Also save the imputation means for inference
    means_path = str(MODEL_DIR / "train_means.pkl")
    with open(means_path, "wb") as f:
        pickle.dump(train_means, f)

    # ── Step 6: Evaluate ──
    results = {}
    for name, X, y in [("Validation", X_val, y_val), ("Test", X_test, y_test)]:
        probs = model.predict(X, num_iteration=model.best_iteration)
        roc_auc = roc_auc_score(y, probs)
        auprc   = average_precision_score(y, probs)

        results[name] = {"ROC-AUC": roc_auc, "AUPRC": auprc}

        log.info(f"\n   📊 {name} Results:")
        log.info(f"      ROC-AUC : {roc_auc:.4f}")
        log.info(f"      AUPRC   : {auprc:.4f}")

    # Feature importance
    log.info("\n   🌟 Top 10 Feature Importances (gain):")
    importance = dict(zip(
        ALL_FEATURES,
        model.feature_importance(importance_type="gain"),
    ))
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:10]:
        log.info(f"      {feat:<25s} {imp:>12.1f}")

    # ── Step 7: Probability calibration (isotonic regression) ──
    log.info("\n   Step 7: Calibrating probabilities (isotonic on val set)...")

    val_probs_raw = model.predict(X_val, num_iteration=model.best_iteration)
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(val_probs_raw, y_val)

    cal_path = str(MODEL_DIR / "probability_calibrator.pkl")
    with open(cal_path, "wb") as f:
        pickle.dump(calibrator, f)
    log.info(f"   Calibrator saved to: {cal_path}")

    cal_probs = calibrator.transform(val_probs_raw)
    log.info(f"   Raw prob mean  : {val_probs_raw.mean():.6f}")
    log.info(f"   Calibrated mean: {cal_probs.mean():.6f}")

    elapsed = time.time() - t0
    log.info(f"\n   Phase 4 complete in {elapsed:.0f}s")

    return model, calibrator, train_means, results
