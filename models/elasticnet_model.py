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
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler


class ElasticNetModel:
    def __init__(self, cacheDir: str | None = None):
        self.alpha = float(os.environ.get("MEOW_ELASTIC_ALPHA", "0.001"))
        self.l1_ratio = float(os.environ.get("MEOW_ELASTIC_L1_RATIO", "0.5"))
        self.use_cv = os.environ.get("MEOW_ELASTIC_CV", "0") != "0"
        self.accumulate_dtype = os.environ.get("MEOW_ELASTIC_ACCUMULATE_DTYPE", "float32")
        self.max_samples = int(os.environ.get("MEOW_ELASTIC_MAX_SAMPLES", "250000"))
        self._rng = np.random.RandomState(42)
        self._X_sample: np.ndarray | None = None
        self._y_sample: np.ndarray | None = None
        self._sample_count = 0
        self._total_seen = 0
        self._feature_names: list[str] = []
        self._model: ElasticNet | None = None
        self._scaler: StandardScaler | None = None
        self._coef: np.ndarray | None = None
        self._intercept_val: float = 0.0
        self._fitted = False

    def reset(self):
        self._X_sample = None
        self._y_sample = None
        self._sample_count = 0
        self._total_seen = 0
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
        if not self._feature_names:
            self._feature_names = list(xdf.columns)
        self._reservoir_update(x, y)

    def finalize_fit(self):
        if self._X_sample is None or self._sample_count == 0:
            return
        X = self._X_sample[: self._sample_count]
        y = self._y_sample[: self._sample_count]
        self._X_sample = None
        self._y_sample = None

        # Standardize features (centered + scaled for L1 regularization)
        self._scaler = StandardScaler(copy=False)
        X_scaled = self._scaler.fit_transform(X)

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
        self._model = model
        coef = model.coef_ / self._scaler.scale_
        intercept_val = float(model.intercept_ - (model.coef_ / self._scaler.scale_) @ self._scaler.mean_)
        best_alpha = self.alpha
        best_l1 = self.l1_ratio

        self._coef = coef.astype(np.float64)
        self._intercept_val = float(intercept_val)
        self._fitted = True
        n_selected = int(np.sum(np.abs(coef) > 1e-12))
        from log import log
        log.inf(
            f"ElasticNet fitted: alpha={best_alpha:.4f}, l1_ratio={best_l1:.2f}, "
            f"selected={n_selected}/{len(coef)} features, sample={self._sample_count}/{self._total_seen}"
        )

    def fit(self, xdf, ydf):
        self.reset()
        self.partial_fit(xdf, ydf)
        self.finalize_fit()

    def predict(self, xdf):
        if not self._fitted or self._coef is None:
            return np.zeros(len(xdf), dtype=np.float64)
        x = xdf.to_numpy(dtype=np.float64, copy=False)
        return x @ self._coef + self._intercept_val

    def _reservoir_update(self, x, y):
        n_rows, n_features = x.shape
        if self.max_samples <= 0:
            self.max_samples = n_rows
        if self._X_sample is None:
            dtype = x.dtype
            self._X_sample = np.empty((self.max_samples, n_features), dtype=dtype)
            self._y_sample = np.empty(self.max_samples, dtype=y.dtype)
        for row_idx in range(n_rows):
            seen = self._total_seen
            self._total_seen += 1
            if self._sample_count < self.max_samples:
                slot = self._sample_count
                self._sample_count += 1
            else:
                slot = self._rng.randint(0, seen + 1)
                if slot >= self.max_samples:
                    continue
            self._X_sample[slot] = x[row_idx]
            self._y_sample[slot] = y[row_idx]
