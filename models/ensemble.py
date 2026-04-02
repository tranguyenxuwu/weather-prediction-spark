"""
Ensemble models for Phase 5: ZINB + LightGBM + Ridge stacking.
"""

import logging

import numpy as np
import pandas as pd
import lightgbm as lgb
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from statsmodels.discrete.discrete_model import NegativeBinomialP
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin

from .sanitize import clamp_predictions

log = logging.getLogger("BottomUpForecast")


class ZINB_Wrapper(BaseEstimator, RegressorMixin):
    """Zero-Inflated Negative Binomial wrapper with NegBin fallback.
    
    Handles:
    - inf/NaN sanitization before StandardScaler (crashes on non-finite)
    - Convergence failure fallback (ZINB → NegBin → zeros)
    - Output clamping (statsmodels can overflow to inf)
    """

    def __init__(self, count_features, inflation_features):
        self.count_features = count_features
        self.inflation_features = inflation_features
        self.model = None
        self.scaler = StandardScaler()
        self.fallback = False

    def fit(self, X, y):
        import warnings
        from statsmodels.tools.sm_exceptions import ConvergenceWarning

        # Sanitize inputs: StandardScaler and statsmodels crash on inf/NaN
        X_count_raw = X[self.count_features].copy()
        X_count_raw = X_count_raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_count = self.scaler.fit_transform(X_count_raw)
        X_infl = (X[self.inflation_features].copy()
                  .replace([np.inf, -np.inf], np.nan).fillna(0.0).values)

        X_count = sm.add_constant(X_count, has_constant='add')
        X_infl = sm.add_constant(X_infl, has_constant='add')

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                zinb = ZeroInflatedNegativeBinomialP(
                    endog=y.values,
                    exog=X_count,
                    exog_infl=X_infl,
                    inflation='logit',
                    p=2
                )
                self.model = zinb.fit(method='bfgs', maxiter=500, disp=False)
        except Exception as e:
            log.warning(f"      [ZINB] Convergence failed ({e}), "
                        f"falling back to NegativeBinomialP")
            self.fallback = True

        if self.fallback:
            try:
                nb = NegativeBinomialP(endog=y.values, exog=X_count, p=2)
                self.model = nb.fit(method='bfgs', maxiter=500, disp=False)
            except Exception as e:
                log.warning(f"      [NB] Convergence also failed ({e}), "
                            f"Wrapper will return zeros")
                self.model = None

        return self

    def predict(self, X):
        X_count_raw = (X[self.count_features].copy()
                       .replace([np.inf, -np.inf], np.nan).fillna(0.0))
        X_count = self.scaler.transform(X_count_raw)
        X_count = sm.add_constant(X_count, has_constant='add')

        if self.model is None:
            return np.zeros(len(X))

        if not self.fallback:
            X_infl = (X[self.inflation_features].copy()
                      .replace([np.inf, -np.inf], np.nan).fillna(0.0).values)
            X_infl = sm.add_constant(X_infl, has_constant='add')
            preds = self.model.predict(exog=X_count, exog_infl=X_infl)
        else:
            preds = self.model.predict(exog=X_count)

        return clamp_predictions(preds)


class Phase5_StackedEnsemble:
    """Heterogeneous stacking: ZINB + Tweedie LightGBM + Ridge meta-learner."""

    def __init__(self, target_col):
        self.target_col = target_col
        self.count_feats = [
            'monthly_SPI_log', 'spi_x_oni', 'cumulative_TCs_YTD',
            'spi_momentum_1m', 'actual_count_lag1',
            'spi_x_month_sin', 'spi_x_month_cos',
        ]
        self.infl_feats = ['month_sin', 'month_cos', 'avg_oni']

        if target_col != "actual_count":
            c = target_col.replace('actual_', '')
            spi_col = f"SPI_{c}"
            self.count_feats = [
                f'{spi_col}_log', 'spi_x_oni', 'cumulative_TCs_YTD',
                'spi_momentum_1m', 'actual_count_lag1',
                'spi_x_month_sin', 'spi_x_month_cos',
            ]
            self.infl_feats = [
                'month_sin', 'month_cos', 'avg_oni', spi_col, 'spi_x_oni',
            ]

        self.model_zinb = ZINB_Wrapper(self.count_feats, self.infl_feats)

        self.model_lgbm = lgb.LGBMRegressor(
            objective='tweedie',
            tweedie_variance_power=1.5,
            num_leaves=6,
            min_child_samples=25,
            learning_rate=0.01,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        self.meta_learner = Ridge(positive=True, alpha=1.0)

    def fit_base_models(self, X_train_full, y_train_full, lgbm_features):
        self.model_zinb.fit(X_train_full, y_train_full)

        X_train_lgb = X_train_full[lgbm_features].copy()
        for col in ['month']:
            X_train_lgb[col] = X_train_lgb[col].astype('category')

        self.model_lgbm.fit(X_train_lgb, y_train_full)
        self.lgbm_features = lgbm_features

    def fit_meta(self, out_of_fold_preds, y_train_cv):
        if len(y_train_cv) > 0:
            self.meta_learner.fit(out_of_fold_preds, y_train_cv)

    def predict(self, X_test):
        pred_zinb = self.model_zinb.predict(X_test)
        # Guard: ensure ZINB predictions are finite before meta-learner
        pred_zinb = clamp_predictions(pred_zinb)

        X_test_lgb = X_test[self.lgbm_features].copy()
        for col in ['month']:
            X_test_lgb[col] = X_test_lgb[col].astype('category')
        pred_lgbm = self.model_lgbm.predict(X_test_lgb)

        meta_features = np.column_stack((pred_zinb, pred_lgbm))

        if hasattr(self.meta_learner, 'coef_'):
            final_preds = self.meta_learner.predict(meta_features)
        else:
            final_preds = pred_lgbm

        return np.maximum(final_preds, 0)


def expanding_window_cv(df, start_cv_year, end_cv_year,
                        target_col, lgbm_features):
    """Walk-forward expanding window CV for the stacked ensemble.
    
    Trains ZINB + LightGBM on expanding training windows, collects
    out-of-fold predictions, then fits the Ridge meta-learner.
    """
    ensemble = Phase5_StackedEnsemble(target_col)
    out_of_fold = []
    y_cv = []

    for year in range(start_cv_year + 1, end_cv_year + 1):
        train_mask = df['year'] < year
        test_mask = df['year'] == year

        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        X_train = df[train_mask]
        y_train = df.loc[train_mask, target_col]
        X_test = df[test_mask]
        y_test = df.loc[test_mask, target_col]

        zinb = ZINB_Wrapper(ensemble.count_feats, ensemble.infl_feats)
        zinb.fit(X_train, y_train)

        lgbm_model = lgb.LGBMRegressor(
            objective='tweedie', tweedie_variance_power=1.5,
            num_leaves=6, min_child_samples=25, learning_rate=0.01,
            n_estimators=300, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )
        X_train_lgb = X_train[lgbm_features].copy()
        for col in ['month']:
            X_train_lgb[col] = X_train_lgb[col].astype('category')
        X_test_lgb = X_test[lgbm_features].copy()
        for col in ['month']:
            X_test_lgb[col] = X_test_lgb[col].astype('category')

        lgbm_model.fit(X_train_lgb, y_train)

        pred_zinb = zinb.predict(X_test)
        # Clamp OOF predictions before they reach Ridge.fit()
        pred_zinb = clamp_predictions(pred_zinb)
        pred_lgbm = lgbm_model.predict(X_test_lgb)

        for i in range(len(pred_zinb)):
            out_of_fold.append([pred_zinb[i], pred_lgbm[i]])
            y_cv.append(y_test.values[i])

    oof_preds = np.array(out_of_fold) if out_of_fold else np.zeros((0, 2))
    y_cv = np.array(y_cv)

    full_train_mask = df['year'] <= end_cv_year
    X_train_full = df[full_train_mask]
    y_train_full = df.loc[full_train_mask, target_col]

    ensemble.fit_base_models(X_train_full, y_train_full, lgbm_features)

    if len(y_cv) > 0:
        ensemble.fit_meta(oof_preds, y_cv)

    return ensemble
