"""Nystroem kernel approximation base model — RBF kernel via random Fourier features.

Replaces linear ElasticNet with kernelized features that capture pairwise feature
interactions automatically. Uses reservoir sampling (like LGBM) for memory control
and closed-form Ridge on the Nystroem-transformed space.

Reference: sklearn.kernel_approximation.Nystroem
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler


class NystroemModel:
    def __init__(self):
        self.max_rows = int(os.environ.get("MEOW_NYSTROEM_MAX_ROWS", "400000"))
        self.n_components = int(os.environ.get("MEOW_NYSTROEM_COMPONENTS", "500"))
        self.gamma = float(os.environ.get("MEOW_NYSTROEM_GAMMA", "0.05"))
        self.ridge_alpha = float(os.environ.get("MEOW_NYSTROEM_RIDGE_ALPHA", "1.0"))
        # Column families to exclude (matching ridge's exclude_families default)
        self.exclude_families = {
            f.strip()
            for f in os.environ.get("MEOW_EXCLUDE_FAMILIES", "cs").split(",")
            if f.strip()
        }
        self._X_reservoir = None
        self._y_reservoir = None
        self._n_accumulated = 0
        self._nystroem = None
        self._scaler = None
        self._coef = None
        self._intercept_val = 0.0
        self._feature_names = None
        self._rng = np.random.RandomState(42)

    def reset(self):
        self._X_reservoir = None
        self._y_reservoir = None
        self._n_accumulated = 0
        self._nystroem = None
        self._scaler = None
        self._coef = None
        self._intercept_val = 0.0
        self._feature_names = None

    def partial_fit(self, xdf, ydf):
        # Select columns, excluding unwanted families
        if self._feature_names is None:
            cols = [c for c in xdf.columns if self._family_of(c) not in self.exclude_families]
            self._feature_names = cols
        x = xdf[self._feature_names].to_numpy(dtype=np.float32)
        y = ydf.to_numpy(dtype=np.float32).ravel()

        n = len(x)
        if self._X_reservoir is None:
            self._X_reservoir = x.copy()
            self._y_reservoir = y.copy()
            self._n_accumulated = n
        else:
            capacity = self._X_reservoir.shape[0]
            if capacity < self.max_rows:
                take = min(n, self.max_rows - capacity)
                self._X_reservoir = np.concatenate([self._X_reservoir, x[:take]], axis=0)
                self._y_reservoir = np.concatenate([self._y_reservoir, y[:take]], axis=0)
            else:
                for i in range(n):
                    j = self._rng.randint(0, self._n_accumulated + i + 1)
                    if j < capacity:
                        self._X_reservoir[j] = x[i]
                        self._y_reservoir[j] = y[i]
            self._n_accumulated += n

    def finalize_fit(self):
        if self._X_reservoir is None or len(self._y_reservoir) < 1000:
            return
        X = self._X_reservoir.astype(np.float64)
        y = self._y_reservoir.astype(np.float64)

        # Standardize before Nystroem (critical for RBF kernel)
        self._scaler = StandardScaler(copy=False)
        X_scaled = self._scaler.fit_transform(X)

        # Fit Nystroem on reservoir
        actual_components = min(self.n_components, len(X_scaled))
        self._nystroem = Nystroem(
            kernel="rbf",
            gamma=self.gamma,
            n_components=actual_components,
            random_state=42,
        )
        Z = self._nystroem.fit_transform(X_scaled)

        # Closed-form Ridge: w = (Z^T Z + alpha*I)^(-1) Z^T y
        ZtZ = Z.T @ Z
        Zty = Z.T @ y
        eye = np.eye(actual_components, dtype=np.float64)
        w = np.linalg.solve(ZtZ + self.ridge_alpha * eye, Zty)

        # Intercept: mean(y) - mean(Z) @ w
        Z_mean = Z.mean(axis=0)
        self._intercept_val = float(y.mean() - Z_mean @ w)
        self._coef = w

        # Free reservoir memory
        self._X_reservoir = None
        self._y_reservoir = None

        from log import log
        log.inf(f"Nystroem fitted: components={actual_components}, gamma={self.gamma}, alpha={self.ridge_alpha}")

    def predict(self, xdf):
        if self._coef is None or self._feature_names is None:
            return np.zeros(len(xdf), dtype=np.float64)
        x = xdf[self._feature_names].to_numpy(dtype=np.float64, copy=False)
        if self._scaler is not None:
            x = self._scaler.transform(x)
        if self._nystroem is not None:
            x = self._nystroem.transform(x)
        return x @ self._coef + self._intercept_val

    @staticmethod
    def _family_of(name):
        if name.endswith("_x_time_sq"):
            return "time_sq_interaction"
        if name.endswith("_x_u_sq"):
            return "u_sq_interaction"
        if name.endswith("_x_time"):
            return "time_interaction"
        if name.endswith("_x_u"):
            return "u_interaction"
        if name.endswith("_rank_cs"):
            return "rank"
        if name.endswith("_cs"):
            return "cs"
        if name.startswith("interval_"):
            return "time_basis"
        return "raw"
