"""CatBoost model with YetiRank ranking objective for blend ensemble.

YetiRank directly optimizes NDCG — a ranking metric that measures how well
predictions order items within groups. This is closer to Pearson correlation
(our eval metric) than MSE regression.

CatBoost's ordered boosting reduces overfitting by using a permutation-based
gradient estimation that doesn't leak target information through the training
set. Groups are (date, interval) pairs — same as Pearson eval groups.
"""
from __future__ import annotations

import os

import numpy as np
import catboost as cb


class CatModel:
    def __init__(self):
        self.max_rows = int(os.environ.get("MEOW_CAT_MAX_ROWS", "500000"))
        self.max_depth = int(os.environ.get("MEOW_CAT_MAX_DEPTH", "6"))
        self.learning_rate = float(os.environ.get("MEOW_CAT_LEARNING_RATE", "0.05"))
        self.n_estimators = int(os.environ.get("MEOW_CAT_N_ESTIMATORS", "200"))
        self.subsample = float(os.environ.get("MEOW_CAT_SUBSAMPLE", "0.8"))
        self.objective = os.environ.get("MEOW_CAT_OBJECTIVE", "YetiRank").strip()
        self.exclude_families = {
            f.strip()
            for f in os.environ.get(
                "MEOW_CAT_EXCLUDE_FAMILIES",
                "cs,time_interaction,u_interaction,time_sq_interaction,u_sq_interaction",
            ).split(",")
            if f.strip()
        }
        self.exclude_patterns = tuple(
            pattern.strip()
            for pattern in os.environ.get(
                "MEOW_CAT_EXCLUDE_PATTERNS",
                "midpx_level_rank_cs,lastpx_level_rank_cs,high_level_rank_cs,"
                "low_level_rank_cs,open_level_rank_cs,bid0_level_rank_cs,ask0_level_rank_cs",
            ).split(",")
            if pattern.strip()
        )
        self._X_reservoir = None
        self._y_reservoir = None
        self._group_ids = None
        self._n_accumulated = 0
        self._model = None
        self._feature_names = None
        self._rng = np.random.RandomState(42)

    def reset(self):
        self._X_reservoir = None
        self._y_reservoir = None
        self._group_ids = None
        self._n_accumulated = 0
        self._model = None
        self._feature_names = None

    def partial_fit(self, xdf, ydf):
        if self._feature_names is None:
            cols = [c for c in xdf.columns if self._keep_column(c)]
            self._feature_names = cols
        x = xdf[self._feature_names].to_numpy(dtype=np.float32)
        y = ydf.to_numpy(dtype=np.float32).ravel()

        # Build group IDs from (date, interval) index levels
        date_vals = ydf.index.get_level_values("date").to_numpy(dtype=np.int64)
        intv_vals = ydf.index.get_level_values("interval").to_numpy(dtype=np.int64)
        group = date_vals * 1000 + intv_vals

        n = len(x)
        if self._X_reservoir is None:
            self._X_reservoir = x.copy()
            self._y_reservoir = y.copy()
            self._group_ids = group.copy()
            self._n_accumulated = n
        else:
            capacity = self._X_reservoir.shape[0]
            if capacity < self.max_rows:
                take = min(n, self.max_rows - capacity)
                self._X_reservoir = np.concatenate([self._X_reservoir, x[:take]], axis=0)
                self._y_reservoir = np.concatenate([self._y_reservoir, y[:take]], axis=0)
                self._group_ids = np.concatenate([self._group_ids, group[:take]], axis=0)
            else:
                for i in range(n):
                    j = self._rng.randint(0, self._n_accumulated + i + 1)
                    if j < capacity:
                        self._X_reservoir[j] = x[i]
                        self._y_reservoir[j] = y[i]
                        self._group_ids[j] = group[i]
            self._n_accumulated += n

    def finalize_fit(self):
        if self._X_reservoir is None or len(self._y_reservoir) < 1000:
            return

        # CatBoost YetiRank: sort by group_id, pass group_ids via Pool
        sort_idx = np.argsort(self._group_ids)
        X_sorted = self._X_reservoir[sort_idx]
        y_sorted = self._y_reservoir[sort_idx]
        gids_sorted = self._group_ids[sort_idx]

        train_pool = cb.Pool(X_sorted, y_sorted, group_id=gids_sorted)

        self._model = cb.CatBoostRegressor(
            iterations=self.n_estimators,
            learning_rate=self.learning_rate,
            depth=self.max_depth,
            loss_function=self.objective,
            subsample=self.subsample,
            random_seed=42,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        )
        self._model.fit(train_pool, silent=True)
        # Free reservoir
        self._X_reservoir = None
        self._y_reservoir = None
        self._group_ids = None

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
