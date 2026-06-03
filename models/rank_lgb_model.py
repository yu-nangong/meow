"""LightGBM with lambdarank objective: learns to rank stocks within (date, interval).

Fundamentally different from the existing MSE-trained LGBModel:
- Objective: lambdarank — directly optimizes for ordering, not absolute values.
- Groups: each (date, interval) is a query group; stocks are items to rank.
- This aligns with the Pearson correlation evaluation metric which measures
  how well predictions rank stocks relative to each other.

References:
- Burges, "From RankNet to LambdaRank to LambdaMART: An Overview", 2010
- LightGBM docs: objective=lambdarank, metric=ndcg
"""
from __future__ import annotations

import os

import numpy as np
import lightgbm as lgb


class RankLGBModel:
    """LGB model trained with lambdarank objective on (date, interval) groups."""

    def __init__(self):
        self.max_rows = int(os.environ.get("MEOW_RANK_LGB_MAX_ROWS", "800000"))
        self.num_leaves = int(os.environ.get("MEOW_RANK_LGB_NUM_LEAVES", "31"))
        self.learning_rate = float(os.environ.get("MEOW_RANK_LGB_LEARNING_RATE", "0.05"))
        self.n_estimators = int(os.environ.get("MEOW_RANK_LGB_N_ESTIMATORS", "200"))
        self.subsample = float(os.environ.get("MEOW_RANK_LGB_SUBSAMPLE", "0.8"))
        self.colsample_bytree = float(os.environ.get("MEOW_RANK_LGB_COLSAMPLE_BYTREE", "0.8"))
        self.min_child_samples = int(os.environ.get("MEOW_RANK_LGB_MIN_CHILD_SAMPLES", "100"))
        self.reg_lambda = float(os.environ.get("MEOW_RANK_LGB_REG_LAMBDA", "1.0"))
        self.exclude_families = {
            f.strip()
            for f in os.environ.get(
                "MEOW_RANK_LGB_EXCLUDE_FAMILIES",
                "cs,time_interaction,u_interaction,time_sq_interaction,u_sq_interaction",
            ).split(",")
            if f.strip()
        }
        self.exclude_patterns = tuple(
            p.strip()
            for p in os.environ.get(
                "MEOW_RANK_LGB_EXCLUDE_PATTERNS",
                "midpx_level_rank_cs,lastpx_level_rank_cs,high_level_rank_cs,"
                "low_level_rank_cs,open_level_rank_cs,bid0_level_rank_cs,ask0_level_rank_cs",
            ).split(",")
            if p.strip()
        )
        self._X_reservoir = None
        self._y_reservoir = None
        self._group_keys_reservoir = None  # (date, interval) tuples
        self._n_accumulated = 0
        self._model = None
        self._feature_names = None
        self._rng = np.random.RandomState(42)

    def reset(self):
        self._X_reservoir = None
        self._y_reservoir = None
        self._group_keys_reservoir = None
        self._n_accumulated = 0
        self._model = None
        self._feature_names = None

    def partial_fit(self, xdf, ydf):
        if self._feature_names is None:
            cols = [c for c in xdf.columns if self._keep_column(c)]
            self._feature_names = cols
        if not self._feature_names:
            return
        x = xdf[self._feature_names].to_numpy(dtype=np.float32)
        y = ydf.to_numpy(dtype=np.float32).ravel()
        idx_frame = ydf.index.to_frame(index=False)
        gk = list(zip(
            idx_frame["date"].to_numpy(dtype=np.int32, copy=False),
            idx_frame["interval"].to_numpy(dtype=np.int32, copy=False),
        ))

        n = len(x)
        if self._X_reservoir is None:
            take = min(n, self.max_rows)
            self._X_reservoir = x[:take].copy()
            self._y_reservoir = y[:take].copy()
            self._group_keys_reservoir = gk[:take]
            self._n_accumulated = n
            return
        capacity = self._X_reservoir.shape[0]
        if capacity < self.max_rows:
            take = min(n, self.max_rows - capacity)
            self._X_reservoir = np.concatenate([self._X_reservoir, x[:take]], axis=0)
            self._y_reservoir = np.concatenate([self._y_reservoir, y[:take]], axis=0)
            self._group_keys_reservoir.extend(gk[:take])
        else:
            for i in range(n):
                j = self._rng.randint(0, self._n_accumulated + i + 1)
                if j < capacity:
                    self._X_reservoir[j] = x[i]
                    self._y_reservoir[j] = y[i]
                    self._group_keys_reservoir[j] = gk[i]
        self._n_accumulated += n

    def finalize_fit(self):
        if self._X_reservoir is None or len(self._y_reservoir) < 1000:
            return
        gk_arr = np.array(self._group_keys_reservoir, dtype=[("date", np.int32), ("interval", np.int32)])
        sort_idx = np.argsort(gk_arr, order=["date", "interval"])
        X_sorted = self._X_reservoir[sort_idx]
        y_sorted = self._y_reservoir[sort_idx]
        gk_sorted = gk_arr[sort_idx]

        groups = []
        prev = None
        count = 0
        for key in gk_sorted:
            key_t = (key["date"], key["interval"])
            if key_t != prev:
                if count > 0:
                    groups.append(count)
                prev = key_t
                count = 1
            else:
                count += 1
        if count > 0:
            groups.append(count)

        params = dict(
            objective="lambdarank",
            metric="ndcg",
            lambdarank_truncation_levels=[10, 30],
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            subsample_freq=1,
            colsample_bytree=self.colsample_bytree,
            min_child_samples=self.min_child_samples,
            reg_lambda=self.reg_lambda,
            verbose=-1,
            random_state=42,
            n_jobs=1,
        )
        train_data = lgb.Dataset(
            X_sorted,
            label=y_sorted,
            group=groups,
            free_raw_data=False,
        )
        self._model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
        )
        self._X_reservoir = None
        self._y_reservoir = None
        self._group_keys_reservoir = None

    def predict(self, xdf):
        if self._model is None or self._feature_names is None or not self._feature_names:
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
