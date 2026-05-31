"""
Ridge Ensemble — Feature Subspace Ridge Averaging for MEOW.

Trains K independent ridge models, each on a random subsample of features.
The ensemble average approximates a nonlinear function through model diversity.
Inspired by Random Forest but using ridge as base learner.

Key advantages for MEOW:
- Stays in numpy/sklearn — no OOM risk
- Each ridge is on ~20 features, much simpler than the 90-feature supermodel
- Ensemble variance reduction captures interaction effects
- Memory: K * (n_subspace^2) stored as XtX matrices, not K * (n_features^2)

Reference: "Ensemble Methods: Foundations and Algorithms" (Zhou, 2012)
"""

from __future__ import annotations

import os
import warnings

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class RidgeEnsemble:
    def __init__(self, cacheDir: str | None = None):
        self.n_estimators = int(os.environ.get("MEOW_ENSEMBLE_N_ESTIMATORS", "25"))
        self.subspace_size = int(os.environ.get("MEOW_ENSEMBLE_SUBSPACE_SIZE", "20"))
        self.subspace_subsample = float(os.environ.get("MEOW_ENSEMBLE_SUBSPACE_SUBSAMPLE", "0.0"))
        self.ridge_alpha = float(os.environ.get("MEOW_ENSEMBLE_ALPHA", "0.001"))
        self.seed = int(os.environ.get("MEOW_ENSEMBLE_SEED", "42"))
        self.max_train_samples = int(os.environ.get("MEOW_ENSEMBLE_MAX_TRAIN_SAMPLES", "0"))
        self.accumulate_dtype = os.environ.get("MEOW_ENSEMBLE_ACCUMULATE_DTYPE", "float32")
        self._X_list: list[np.ndarray] = []
        self._y_list: list[np.ndarray] = []
        self._feature_names: list[str] = []
        self._subspaces: list[list[int]] = []  # list of column index lists
        self._models: list[tuple[np.ndarray, float]] = []  # list of (coef, intercept)
        self._fitted = False
        self._rng = np.random.RandomState(self.seed)

    def reset(self):
        self._X_list = []
        self._y_list = []
        self._feature_names = []
        self._subspaces = []
        self._models = []
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
        X = np.concatenate(self._X_list, axis=0).astype(np.float64)
        y = np.concatenate(self._y_list, axis=0).astype(np.float64)
        self._X_list = []
        self._y_list = []

        n_samples, n_features = X.shape

        # Optional training sample subsampling for speed
        if self.max_train_samples > 0 and n_samples > self.max_train_samples:
            idx = self._rng.choice(n_samples, self.max_train_samples, replace=False)
            X, y = X[idx], y[idx]
            n_samples = len(y)

        # Determine subspace size
        subspace_size = min(self.subspace_size, n_features)
        if self.subspace_subsample > 0:
            subspace_size = max(3, int(n_features * self.subspace_subsample))

        # Generate random feature subspaces and fit ridge models
        all_indices = np.arange(n_features)
        self._subspaces = []

        for i in range(self.n_estimators):
            np.random.RandomState(self.seed + i * 7)
            if subspace_size >= n_features:
                col_idx = all_indices.copy()
            else:
                col_idx = self._rng.choice(n_features, subspace_size, replace=False)
                col_idx.sort()
            self._subspaces.append(col_idx.tolist())

            # Extract subspace
            X_sub = X[:, col_idx]

            # Standardize within subspace
            mean_x = X_sub.mean(axis=0)
            std_x = X_sub.std(axis=0).clip(min=1e-12)
            X_sub_scaled = (X_sub - mean_x[None, :]) / std_x[None, :]

            # Solve ridge: (X^T X + alpha*I) coef = X^T y
            XtX = X_sub_scaled.T @ X_sub_scaled
            Xty = X_sub_scaled.T @ y
            ridge_diag = np.full(subspace_size, self.ridge_alpha)
            try:
                coef_scaled = np.linalg.solve(XtX + np.diag(ridge_diag), Xty)
            except np.linalg.LinAlgError:
                coef_scaled = np.linalg.lstsq(XtX + np.diag(ridge_diag), Xty, rcond=None)[0]

            # Transform coefficients back to original scale
            coef = coef_scaled / std_x
            intercept_val = float(y.mean() - (coef_scaled / std_x) @ mean_x)

            self._models.append((coef.astype(np.float64), float(intercept_val)))

        self._fitted = True
        from log import log
        log.inf(f"RidgeEnsemble fitted: {len(self._models)} estimators, subspace={subspace_size}/{n_features}")

    def fit(self, xdf, ydf):
        self.reset()
        self.partial_fit(xdf, ydf)
        self.finalize_fit()

    def predict(self, xdf) -> np.ndarray:
        if not self._fitted or not self._models:
            return np.zeros(len(xdf), dtype=np.float64)
        x = xdf.to_numpy(dtype=np.float64, copy=False)
        n = len(x)
        preds = np.zeros((n, len(self._models)), dtype=np.float64)
        for i, (coef, intercept) in enumerate(self._models):
            col_idx = self._subspaces[i]
            preds[:, i] = x[:, col_idx] @ coef + intercept
        # Ensemble mean
        return preds.mean(axis=1)
