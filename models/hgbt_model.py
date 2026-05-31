"""
HistGradientBoosting primary model — replaces 2-stage (ElasticNet + interval residual).

Why HGBT for this task:
- Naturally captures nonlinear relationships ridge cannot
- Histogram binning keeps memory O(bins*n_features) not O(n_samples*n_features)
- Ordered boosting reduces overfitting on noisy financial data
- No per-interval post-processing needed — interval effects learned directly
- Single-stage avoids the 2-stage competition that capped Linear+Interval at 0.071

Memory strategy: Accumulate training data in float32, optionally subsample
if total rows exceed a threshold.
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


class HgbtModel:
    def __init__(self):
        self.max_iter = int(os.environ.get("MEOW_HGBT_MAX_ITER", "500"))
        self.learning_rate = float(os.environ.get("MEOW_HGBT_LR", "0.1"))
        self.max_leaf_nodes = int(os.environ.get("MEOW_HGBT_MAX_LEAF_NODES", "31"))
        self.min_samples_leaf = int(os.environ.get("MEOW_HGBT_MIN_SAMPLES_LEAF", "100"))
        self.l2_regularization = float(os.environ.get("MEOW_HGBT_L2", "0.3"))
        self.max_bins = int(os.environ.get("MEOW_HGBT_MAX_BINS", "255"))
        self.max_samples = int(os.environ.get("MEOW_HGBT_MAX_SAMPLES", "800000"))
        self.subsample_seed = int(os.environ.get("MEOW_HGBT_SUBSAMPLE_SEED", "42"))
        self.early_stopping_rounds = int(os.environ.get("MEOW_HGBT_EARLY_STOPPING", "10"))
        self.validation_split = float(os.environ.get("MEOW_HGBT_VALIDATION_SPLIT", "0.15"))
        self._X_list: list[np.ndarray] = []
        self._y_list: list[np.ndarray] = []
        self._n_rows = 0
        self._model: HistGradientBoostingRegressor | None = None
        self._feature_names: list[str] = []
        self._fitted = False

    def reset(self):
        self._X_list = []
        self._y_list = []
        self._n_rows = 0
        self._model = None
        self._feature_names = []
        self._fitted = False

    def partial_fit(self, xdf, ydf):
        x = xdf.to_numpy(dtype=np.float32, copy=False)
        y = ydf.to_numpy(dtype=np.float64).ravel()
        n = len(x)
        max_chunk = int(os.environ.get("MEOW_HGBT_MAX_CHUNK", "150000"))
        if n > max_chunk:
            rng = np.random.RandomState(self.subsample_seed + len(self._X_list))
            idx = rng.choice(n, max_chunk, replace=False)
            x = x[idx]
            y = y[idx]
        self._X_list.append(x)
        self._y_list.append(y.astype(np.float32))
        self._n_rows += len(y)
        if not self._feature_names:
            self._feature_names = list(xdf.columns)

    def finalize_fit(self):
        if not self._X_list:
            self._model = HistGradientBoostingRegressor(n_iter_no_change=0)
            self._model.fit(np.zeros((1, 1)), np.zeros(1))
            self._fitted = True
            return
        X = np.concatenate(self._X_list, axis=0)
        y = np.concatenate(self._y_list, axis=0).ravel()
        self._X_list = []
        self._y_list = []
        n = len(y)
        if n > self.max_samples:
            rng = np.random.RandomState(self.subsample_seed)
            idx = rng.choice(n, self.max_samples, replace=False)
            X = X[idx]
            y = y[idx]
        # Use early_stopping with internal validation split
        use_early_stop = self.early_stopping_rounds > 0 and n > 10000
        self._model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            max_bins=self.max_bins,
            early_stopping=use_early_stop,
            n_iter_no_change=self.early_stopping_rounds if use_early_stop else None,
            validation_split=None if not use_early_stop else self.validation_split,
            scoring="loss",
            verbose=0,
            random_state=self.subsample_seed + 1,
        )
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, xdf):
        x = xdf[self._feature_names].to_numpy(dtype=np.float32, copy=False)
        return self._model.predict(x).astype(np.float64)
