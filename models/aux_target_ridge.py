from __future__ import annotations

import os

import numpy as np


class AuxTargetRidge:
    """Cheap ridge arm for interval-normalized auxiliary targets."""

    def __init__(self):
        self.alpha = float(os.environ.get("MEOW_AUX_TARGET_ALPHA", "0.02"))
        self.include_patterns = tuple(
            pattern.strip()
            for pattern in os.environ.get(
                "MEOW_AUX_TARGET_INCLUDE_PATTERNS",
                "_rank_cs,_cs,_zs,interval_,time_sin_,time_cos_",
            ).split(",")
            if pattern.strip()
        )
        self.exclude_patterns = tuple(
            pattern.strip()
            for pattern in os.environ.get(
                "MEOW_AUX_TARGET_EXCLUDE_PATTERNS",
                "_x_time,_x_u",
            ).split(",")
            if pattern.strip()
        )
        self._feature_names = None
        self._sum_x = None
        self._sum_x2 = None
        self._sum_y = 0.0
        self._n_rows = 0
        self._xtx = None
        self._xty = None
        self._coef = None
        self._intercept = 0.0

    def reset(self):
        self._feature_names = None
        self._sum_x = None
        self._sum_x2 = None
        self._sum_y = 0.0
        self._n_rows = 0
        self._xtx = None
        self._xty = None
        self._coef = None
        self._intercept = 0.0

    def partial_fit(self, xdf, y):
        if self._feature_names is None:
            self._feature_names = [name for name in xdf.columns if self._keep_column(name)]
        if not self._feature_names:
            return
        x = xdf.loc[:, self._feature_names].to_numpy(dtype=np.float64, copy=False)
        y = np.asarray(y, dtype=np.float64).ravel()
        if self._xtx is None:
            n_features = x.shape[1]
            self._sum_x = np.zeros(n_features, dtype=np.float64)
            self._sum_x2 = np.zeros(n_features, dtype=np.float64)
            self._xtx = np.zeros((n_features, n_features), dtype=np.float64)
            self._xty = np.zeros(n_features, dtype=np.float64)
        self._sum_x += x.sum(axis=0)
        self._sum_x2 += np.square(x).sum(axis=0)
        self._sum_y += y.sum()
        self._n_rows += len(y)
        self._xtx += x.T @ x
        self._xty += x.T @ y

    def finalize_fit(self):
        if self._xtx is None or self._n_rows == 0:
            return
        mean_x = self._sum_x / self._n_rows
        var_x = self._sum_x2 / self._n_rows - np.square(mean_x)
        scale_x = np.sqrt(np.maximum(var_x, 1e-12))
        inv_scale = 1.0 / scale_x
        centered_xtx = self._xtx - self._n_rows * np.outer(mean_x, mean_x)
        centered_xty = self._xty - mean_x * self._sum_y
        ztz = centered_xtx * np.outer(inv_scale, inv_scale)
        zty = centered_xty * inv_scale
        coef_scaled = np.linalg.solve(ztz + self.alpha * np.eye(len(zty), dtype=np.float64), zty)
        coef = coef_scaled * inv_scale
        mean_y = self._sum_y / self._n_rows
        self._coef = coef
        self._intercept = float(mean_y - mean_x @ coef)

    def predict(self, xdf):
        if self._coef is None or not self._feature_names:
            return np.zeros(len(xdf), dtype=np.float64)
        x = xdf.loc[:, self._feature_names].to_numpy(dtype=np.float64, copy=False)
        return x @ self._coef + self._intercept

    def _keep_column(self, name):
        if self.exclude_patterns and any(pattern in name for pattern in self.exclude_patterns):
            return False
        if not self.include_patterns:
            return True
        return any(pattern in name for pattern in self.include_patterns)
