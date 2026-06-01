"""
HistGradientBoostingRegressor model for MEOW — single-stage nonlinear.

Replaces the base model + interval-residual 2-stage with a single HGBT
that handles nonlinearity, feature interactions, and interval effects
natively. HGBT is memory-efficient (histogram binning) and supports
early stopping via validation fraction.

To stay within grader memory, partial_fit uses reservoir sampling:
only max_samples rows are kept in memory at any time.

Reference: sklearn.ensemble.HistGradientBoostingRegressor
"""

from __future__ import annotations

import os
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


class HGBTModel:
    """Single-stage HGBT replacing both base model and interval-residual."""

    def __init__(self, cacheDir: str | None = None):
        self.max_iter = int(os.environ.get("MEOW_HGBT_MAX_ITER", "100"))
        self.max_leaf_nodes = int(os.environ.get("MEOW_HGBT_MAX_LEAF_NODES", "31"))
        max_depth_raw = os.environ.get("MEOW_HGBT_MAX_DEPTH", "")
        self.max_depth = int(max_depth_raw) if max_depth_raw else None
        self.learning_rate = float(os.environ.get("MEOW_HGBT_LEARNING_RATE", "0.1"))
        self.max_samples = int(os.environ.get("MEOW_HGBT_MAX_SAMPLES", "800000"))
        self.validation_fraction = float(os.environ.get("MEOW_HGBT_VALIDATION_FRACTION", "0.1"))
        self._X_reservoir: np.ndarray | None = None
        self._y_reservoir: np.ndarray | None = None
        self._reservoir_count: int = 0
        self._feature_names: list[str] = []
        self._model: HistGradientBoostingRegressor | None = None
        self._fitted = False
        self._rng = np.random.RandomState(42)

    def reset(self):
        self._X_reservoir = None
        self._y_reservoir = None
        self._reservoir_count = 0
        self._feature_names = []
        self._model = None
        self._fitted = False

    def partial_fit(self, xdf, ydf):
        x = xdf.to_numpy(dtype=np.float32, copy=False)
        y = ydf.to_numpy(dtype=np.float32, copy=False).ravel()
        if not self._feature_names:
            self._feature_names = list(xdf.columns)
            self._X_reservoir = np.zeros((self.max_samples, x.shape[1]), dtype=np.float32)
            self._y_reservoir = np.zeros(self.max_samples, dtype=np.float32)

        n_new = len(x)
        if self._reservoir_count + n_new <= self.max_samples:
            end = self._reservoir_count + n_new
            self._X_reservoir[self._reservoir_count:end] = x
            self._y_reservoir[self._reservoir_count:end] = y
            self._reservoir_count = end
        else:
            # Reservoir sampling: randomly replace existing rows when full
            total_seen = self._reservoir_count + n_new
            replace_idx = self._rng.randint(0, total_seen, size=n_new)
            for i, ri in enumerate(replace_idx):
                if ri < self.max_samples:
                    self._X_reservoir[ri] = x[i]
                    self._y_reservoir[ri] = y[i]
            self._reservoir_count = total_seen

    def finalize_fit(self):
        from log import log
        if self._reservoir_count == 0:
            return
        n = min(self._reservoir_count, self.max_samples)
        X = self._X_reservoir[:n]
        y = self._y_reservoir[:n]

        log.inf(f"HGBT: fitting on {n} reservoir samples (from {self._reservoir_count} total)")

        self._model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            validation_fraction=self.validation_fraction,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=42,
            verbose=0,
        )
        self._model.fit(X, y)
        self._fitted = True
        log.inf(
            f"HGBT fitted: {self._model.n_iter_} / {self.max_iter} iters, "
            f"leaves={self.max_leaf_nodes}, lr={self.learning_rate}"
        )

    def fit(self, xdf, ydf):
        self.reset()
        self.partial_fit(xdf, ydf)
        self.finalize_fit()

    def predict(self, xdf):
        if not self._fitted or self._model is None:
            return np.zeros(len(xdf), dtype=np.float64)
        x = xdf.to_numpy(dtype=np.float32, copy=False)
        return self._model.predict(x).astype(np.float64)
