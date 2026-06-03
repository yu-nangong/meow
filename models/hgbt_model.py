"""HistGradientBoostingRegressor model for ensemble blend.
Different boosting algorithm than LightGBM — may capture complementary patterns.
"""
from __future__ import annotations

import os
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


class HGBTModel:
    def __init__(self):
        self.max_rows = int(os.environ.get("MEOW_HGBT_MAX_ROWS", "400000"))
        self.max_iter = int(os.environ.get("MEOW_HGBT_MAX_ITER", "200"))
        self.learning_rate = float(os.environ.get("MEOW_HGBT_LEARNING_RATE", "0.05"))
        self.max_depth = int(os.environ.get("MEOW_HGBT_MAX_DEPTH", "None") or "0") or None
        self.min_samples_leaf = int(os.environ.get("MEOW_HGBT_MIN_SAMPLES_LEAF", "20"))
        self.l2_regularization = float(os.environ.get("MEOW_HGBT_L2_REG", "0.0"))
        self.exclude_families = {
            f.strip()
            for f in os.environ.get(
                "MEOW_HGBT_EXCLUDE_FAMILIES",
                "cs,time_interaction,u_interaction,time_sq_interaction,u_sq_interaction",
            ).split(",")
            if f.strip()
        }
        self.exclude_patterns = tuple(
            pattern.strip()
            for pattern in os.environ.get(
                "MEOW_HGBT_EXCLUDE_PATTERNS",
                "midpx_level_rank_cs,lastpx_level_rank_cs,high_level_rank_cs,"
                "low_level_rank_cs,open_level_rank_cs,bid0_level_rank_cs,ask0_level_rank_cs",
            ).split(",")
            if pattern.strip()
        )
        self._X_reservoir = None
        self._y_reservoir = None
        self._n_accumulated = 0
        self._model = None
        self._feature_names = None
        self._rng = np.random.RandomState(43)

    def reset(self):
        self._X_reservoir = None
        self._y_reservoir = None
        self._n_accumulated = 0
        self._model = None
        self._feature_names = None

    def partial_fit(self, xdf, ydf):
        if self._feature_names is None:
            cols = [c for c in xdf.columns if self._keep_column(c)]
            self._feature_names = cols
        x = xdf[self._feature_names].to_numpy(dtype=np.float32)
        y = ydf.to_numpy(dtype=np.float64).ravel()

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
        self._model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=False,
            random_state=43,
            verbose=0,
        )
        self._model.fit(self._X_reservoir, self._y_reservoir)

    def predict(self, xdf):
        if self._model is None or self._feature_names is None:
            return np.zeros(len(xdf), dtype=np.float64)
        x = xdf[self._feature_names].to_numpy(dtype=np.float32)
        return self._model.predict(x).astype(np.float64)

    def _keep_column(self, name):
        if self.exclude_patterns and any(pattern in name for pattern in self.exclude_patterns):
            return False
        return self._family_of(name) not in self.exclude_families

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
