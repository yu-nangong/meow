"""
ElasticNet base model for MEOW — replaces Ridge with L1+L2 regularization.

ElasticNet selects features via L1 sparsity while retaining Ridge's grouping
via L2. For financial data where most predictors are noise, feature selection
should improve out-of-sample Pearson correlation.

Uses chunked accumulation (like MeowModel) + a final ElasticNetCV fit on
accumulated data. The accumulated X,y are kept at coarsened precision to
stay within grader memory.

Reference: sklearn.linear_model.ElasticNet / ElasticNetCV
"""

from __future__ import annotations

import os
import warnings

import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler


class ElasticNetModel:
    def __init__(self, cacheDir: str | None = None):
        self.alpha = float(os.environ.get("MEOW_ELASTIC_ALPHA", "0.001"))
        self.l1_ratio = float(os.environ.get("MEOW_ELASTIC_L1_RATIO", "0.5"))
        self.use_cv = os.environ.get("MEOW_ELASTIC_CV", "1") != "0"
        if self.use_cv:
            cv_alphas_str = os.environ.get("MEOW_ELASTIC_CV_ALPHAS", "0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2")
            self.cv_alphas = [float(a) for a in cv_alphas_str.split(",") if a.strip()]
            cv_l1_str = os.environ.get("MEOW_ELASTIC_CV_L1_RATIO", "0.1,0.3,0.5,0.7,0.9,0.95,0.99")
            self.cv_l1_ratios = [float(l1) for l1 in cv_l1_str.split(",") if l1.strip()]
        self.accumulate_dtype = os.environ.get("MEOW_ELASTIC_ACCUMULATE_DTYPE", "float32")
        self.max_samples = int(os.environ.get("MEOW_ELASTIC_MAX_SAMPLES", "0"))
        self._X_list: list[np.ndarray] = []
        self._y_list: list[np.ndarray] = []
        self._feature_names: list[str] = []
        self._model: ElasticNetCV | None = None
        self._scaler: StandardScaler | None = None
        self._coef: np.ndarray | None = None
        self._intercept_val: float = 0.0
        self._fitted = False

    def reset(self):
        self._X_list = []
        self._y_list = []
        self._feature_names = []
        self._model = None
        self._scaler = None
        self._coef = None
        self._intercept_val = 0.0
        self._fitted = False

    def partial_fit(self, xdf, ydf):
        dtype = np.float32 if self.accumulate_dtype == "float32" else np.float64
        x = xdf.to_numpy(dtype=dtype, copy=False)
        y = ydf.to_numpy(dtype=dtype, copy=False).ravel()
        self._X_list.append(x)
        self._y_list.append(y)
        if not self._feature_names:
            self._feature_names = list(xdf.columns)

    def finalize_fit(self):
        if not self._X_list:
            return
        X = np.concatenate(self._X_list, axis=0)
        y = np.concatenate(self._y_list, axis=0)
        self._X_list = []
        self._y_list = []

        n = len(X)
        if self.max_samples > 0 and n > self.max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, self.max_samples, replace=False)
            X, y = X[idx], y[idx]

        # Standardize features (centered + scaled for L1 regularization)
        self._scaler = StandardScaler(copy=False)
        X_scaled = self._scaler.fit_transform(X)

        if self.use_cv:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = ElasticNetCV(
                    alphas=self.cv_alphas,
                    l1_ratio=self.cv_l1_ratios,
                    fit_intercept=True,
                    max_iter=2000,
                    tol=1e-4,
                    cv=3,
                    n_jobs=1,
                    random_state=42,
                    selection="random",
                )
                self._model.fit(X_scaled, y)
            best_alpha = self._model.alpha_
            best_l1 = self._model.l1_ratio_
            coef_scaled = self._model.coef_
            intercept_raw = float(self._model.intercept_)
            # Transform coefficients back to original scale
            coef = coef_scaled / self._scaler.scale_
            intercept_val = float(intercept_raw - (coef_scaled / self._scaler.scale_) @ self._scaler.mean_)
        else:
            # Fixed ElasticNet (no CV)
            from sklearn.linear_model import ElasticNet

            model = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=True,
                max_iter=2000,
                tol=1e-4,
                random_state=42,
                selection="random",
            )
            model.fit(X_scaled, y)
            coef = model.coef_ / self._scaler.scale_
            intercept_val = float(model.intercept_ - (model.coef_ / self._scaler.scale_) @ self._scaler.mean_)
            best_alpha = self.alpha
            best_l1 = self.l1_ratio

        self._coef = coef.astype(np.float64)
        self._intercept_val = float(intercept_val)
        self._fitted = True
        n_selected = int(np.sum(np.abs(coef) > 1e-12))
        from log import log
        log.inf(f"ElasticNet fitted: alpha={best_alpha:.4f}, l1_ratio={best_l1:.2f}, selected={n_selected}/{len(coef)} features")

    def fit(self, xdf, ydf):
        self.reset()
        self.partial_fit(xdf, ydf)
        self.finalize_fit()

    def predict(self, xdf):
        if not self._fitted or self._coef is None:
            return np.zeros(len(xdf), dtype=np.float64)
        x = xdf.to_numpy(dtype=np.float64, copy=False)
        return x @ self._coef + self._intercept_val
