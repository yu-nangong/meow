"""XGBoost model with pairwise ranking objective for blend ensemble.

Uses rank:pairwise to directly optimize for within-group ordering,
which is what Pearson correlation fundamentally measures.
Groups are (date, interval) pairs.
"""
from __future__ import annotations

import os

import numpy as np
import xgboost as xgb


class XGBModel:
    def __init__(self):
        self.max_rows = int(os.environ.get("MEOW_XGB_MAX_ROWS", "500000"))
        self.max_depth = int(os.environ.get("MEOW_XGB_MAX_DEPTH", "6"))
        self.learning_rate = float(os.environ.get("MEOW_XGB_LEARNING_RATE", "0.05"))
        self.n_estimators = int(os.environ.get("MEOW_XGB_N_ESTIMATORS", "201"))
        self.subsample = float(os.environ.get("MEOW_XGB_SUBSAMPLE", "0.8"))
        self.colsample_bytree = float(os.environ.get("MEOW_XGB_COLSAMPLE_BYTREE", "0.8"))
        self.objective = os.environ.get("MEOW_XGB_OBJECTIVE", "reg:squarederror").strip()
        self.exclude_families = {
            f.strip()
            for f in os.environ.get(
                "MEOW_XGB_EXCLUDE_FAMILIES",
                "cs,time_interaction,u_interaction,time_sq_interaction,u_sq_interaction",
            ).split(",")
            if f.strip()
        }
        self.exclude_patterns = tuple(
            pattern.strip()
            for pattern in os.environ.get(
                "MEOW_XGB_EXCLUDE_PATTERNS",
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

        # Track group IDs for ranking objectives
        if self.objective.startswith("rank:"):
            date_vals = ydf.index.get_level_values("date").to_numpy(dtype=np.int64)
            intv_vals = ydf.index.get_level_values("interval").to_numpy(dtype=np.int64)
            group = date_vals * 1000 + intv_vals
        else:
            group = None

        n = len(x)
        if self._X_reservoir is None:
            self._X_reservoir = x.copy()
            self._y_reservoir = y.copy()
            if group is not None:
                self._group_ids = group.copy()
            self._n_accumulated = n
        else:
            capacity = self._X_reservoir.shape[0]
            if capacity < self.max_rows:
                take = min(n, self.max_rows - capacity)
                self._X_reservoir = np.concatenate([self._X_reservoir, x[:take]], axis=0)
                self._y_reservoir = np.concatenate([self._y_reservoir, y[:take]], axis=0)
                if group is not None:
                    self._group_ids = np.concatenate([self._group_ids, group[:take]], axis=0)
            else:
                for i in range(n):
                    j = self._rng.randint(0, self._n_accumulated + i + 1)
                    if j < capacity:
                        self._X_reservoir[j] = x[i]
                        self._y_reservoir[j] = y[i]
                        if group is not None:
                            self._group_ids[j] = group[i]
            self._n_accumulated += n

    def finalize_fit(self):
        if self._X_reservoir is None or len(self._y_reservoir) < 1000:
            return

        if self._group_ids is not None:
            sort_idx = np.argsort(self._group_ids)
            X_sorted = self._X_reservoir[sort_idx]
            y_sorted = self._y_reservoir[sort_idx]
            gids_sorted = self._group_ids[sort_idx]
            _, group_sizes = np.unique(gids_sorted, return_counts=True)

            # XGBoost needs group boundaries: cumulative sum of sizes, excluding last
            # Use group sizes directly (XGBoost 3.x set_group expects sizes, not boundaries)
            group_sizes_list = group_sizes.tolist()

            dtrain = xgb.DMatrix(
                X_sorted, label=y_sorted,
            )
            dtrain.set_group(group_sizes_list)
        else:
            dtrain = xgb.DMatrix(self._X_reservoir, label=self._y_reservoir)

        params = dict(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective=self.objective,
            tree_method="hist",
            random_state=42,
            nthread=1,
            verbosity=0,
        )

        self._model = xgb.train(
            params, dtrain, num_boost_round=self.n_estimators,
        )

    def predict(self, xdf):
        if self._model is None or self._feature_names is None:
            return np.zeros(len(xdf), dtype=np.float64)
        x = xdf[self._feature_names].to_numpy(dtype=np.float32)
        dtest = xgb.DMatrix(x)
        return self._model.predict(dtest).astype(np.float64)

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
