"""
Gradient-boosted residual correction on top of ridge.

Trains a HistGradientBoostingRegressor on a SUBSAMPLE of training data
to correct ridge residuals. Subsampling avoids OOM (previous GBRT attempt
concatenated all training data and crashed).

Key: The GBRT only sees the residual after ridge + interval-residual,
so it learns nonlinear patterns the linear model missed.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


class GbrtResidual:
    """
    Second-stage nonlinear residual correction.

    Trained on a subsample of (features, ridge_residual). Predicts a correction
    added on top of the ridge + interval-residual forecast.

    partial_fit: accumulates a random subsample of (X, residual) from each chunk.
    finalize_fit: fits GBRT on accumulated subsample.
    predict: returns correction to add.
    """

    def __init__(self):
        self._X_list: list[np.ndarray] = []
        self._y_list: list[np.ndarray] = []
        self._model: HistGradientBoostingRegressor | None = None
        self._feature_names: list[str] = []
        self._fitted = False

        self.max_samples_per_chunk = int(os.environ.get("MEOW_GBRT_RESIDUAL_MAX_SAMPLES", "50000"))
        self.max_iter = int(os.environ.get("MEOW_GBRT_RESIDUAL_MAX_ITER", "300"))
        self.learning_rate = float(os.environ.get("MEOW_GBRT_RESIDUAL_LR", "0.05"))
        self.max_leaf_nodes = int(os.environ.get("MEOW_GBRT_RESIDUAL_MAX_LEAF_NODES", "64"))
        self.min_samples_leaf = int(os.environ.get("MEOW_GBRT_RESIDUAL_MIN_SAMPLES_LEAF", "100"))
        self.l2_regularization = float(os.environ.get("MEOW_GBRT_RESIDUAL_L2", "0.1"))

    def reset(self):
        self._X_list = []
        self._y_list = []
        self._model = None
        self._feature_names = []
        self._fitted = False

    def partial_fit(self, xdf, residual):
        x = xdf.to_numpy(dtype=np.float32, copy=False)
        y = np.asarray(residual, dtype=np.float32).ravel()
        n = len(x)
        if n > self.max_samples_per_chunk:
            idx = np.random.RandomState(42).choice(n, self.max_samples_per_chunk, replace=False)
            x, y = x[idx], y[idx]
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

        # Sub-sample to a hard max if partial-fits still overshoot
        hard_max = int(os.environ.get("MEOW_GBRT_RESIDUAL_HARD_MAX", "300000"))
        if len(X) > hard_max:
            idx = np.random.RandomState(42).choice(len(X), hard_max, replace=False)
            X, y = X[idx], y[idx]

        self._model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=False,
            random_state=42,
        )
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, xdf):
        if self._model is None or not self._fitted:
            return np.zeros(len(xdf), dtype=np.float64)
        x = xdf.to_numpy(dtype=np.float32, copy=False)
        return self._model.predict(x).astype(np.float64)
