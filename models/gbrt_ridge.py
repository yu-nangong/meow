"""
HistGradientBoosting-based base model.

Replaces Ridge with a gradient-boosted regression tree ensemble.
Captures nonlinear interactions and feature hierarchies that ridge cannot express.
Accumulates data across chunks, fits once in finalize_fit.

Reference:
    sklearn.ensemble.HistGradientBoostingRegressor — fast, CPU-native, handles
    high-dimensional tabular data with built-in binning.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


class GbrtRidge:
    """
    Drop-in for MeowModel (ridge) that uses HistGradientBoosting.

    partial_fit: accumulates X, y from each chunk.
    finalize_fit: fits the GBRT on accumulated data, then discards X, y.
    """

    def __init__(self, cacheDir: str | None = None):
        self._X_chunks: list[np.ndarray] = []
        self._y_chunks: list[np.ndarray] = []
        self._feature_names: list[str] = []
        self._model: HistGradientBoostingRegressor | None = None
        self._fitted = False

        # Config from env
        self.max_iter = int(os.environ.get("MEOW_GBRT_MAX_ITER", "300"))
        self.learning_rate = float(os.environ.get("MEOW_GBRT_LR", "0.1"))
        self.max_leaf_nodes = int(os.environ.get("MEOW_GBRT_MAX_LEAF_NODES", "31"))
        self.min_samples_leaf = int(os.environ.get("MEOW_GBRT_MIN_SAMPLES_LEAF", "20"))
        self.l2_regularization = float(os.environ.get("MEOW_GBRT_L2", "0.0"))
        self.max_bins = int(os.environ.get("MEOW_GBRT_MAX_BINS", "255"))

    def reset(self):
        self._X_chunks = []
        self._y_chunks = []
        self._feature_names = []
        self._model = None
        self._fitted = False

    def partial_fit(self, xdf, ydf):
        x = xdf.to_numpy(dtype=np.float32, copy=False)
        y = ydf.to_numpy(dtype=np.float32, copy=False).ravel()
        self._X_chunks.append(x)
        self._y_chunks.append(y)
        if not self._feature_names:
            self._feature_names = list(xdf.columns)

    def finalize_fit(self):
        if not self._X_chunks:
            return
        X = np.concatenate(self._X_chunks, axis=0)
        y = np.concatenate(self._y_chunks, axis=0)
        self._X_chunks = []
        self._y_chunks = []

        self._model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            max_bins=self.max_bins,
            early_stopping=False,
            random_state=42,
        )
        self._model.fit(X, y)
        self._fitted = True

    def fit(self, xdf, ydf):
        self.reset()
        self.partial_fit(xdf, ydf)
        self.finalize_fit()

    def predict(self, xdf):
        if self._model is None or not self._fitted:
            return np.zeros(len(xdf), dtype=np.float64)
        x = xdf.to_numpy(dtype=np.float32, copy=False)
        return self._model.predict(x).astype(np.float64)
