"""LightGBM base model with reservoir-sampled chunked training.
Supports ranking objectives (lambdarank) with group-aware sampling.
"""
from __future__ import annotations

import os

import numpy as np
import lightgbm as lgb


class LGBModel:
    def __init__(self):
        self.max_rows = int(os.environ.get("MEOW_LGB_MAX_ROWS", "800000"))
        self.num_leaves = int(os.environ.get("MEOW_LGB_NUM_LEAVES", "31"))
        self.learning_rate = float(os.environ.get("MEOW_LGB_LEARNING_RATE", "0.05"))
        self.n_estimators = int(os.environ.get("MEOW_LGB_N_ESTIMATORS", "200"))
        self.objective = os.environ.get("MEOW_LGB_OBJECTIVE", "lambdarank").strip()
        self.lambdarank_truncation = int(os.environ.get("MEOW_LGB_LAMBDARANK_TRUNCATION", "30"))
        self.extra_trees = os.environ.get("MEOW_LGB_EXTRA_TREES", "0") != "0"
        self.subsample = float(os.environ.get("MEOW_LGB_SUBSAMPLE", "0.8"))
        self.colsample_bytree = float(os.environ.get("MEOW_LGB_COLSAMPLE_BYTREE", "0.8"))
        self.min_child_samples = int(os.environ.get("MEOW_LGB_MIN_CHILD_SAMPLES", "100"))
        self.reg_lambda = float(os.environ.get("MEOW_LGB_REG_LAMBDA", "1.0"))
        self.is_ranking = self.objective in ("lambdarank", "rank_xendcg")
        # For ranking, give LGB access to more feature families since ranking
        # uses signal differently than MSE regression.
        _default_exclude = (
            "time_sq_interaction,u_sq_interaction"
            if self.is_ranking
            else "cs,time_interaction,u_interaction,time_sq_interaction,u_sq_interaction"
        )
        self.exclude_families = {
            f.strip()
            for f in os.environ.get(
                "MEOW_LGB_EXCLUDE_FAMILIES", _default_exclude
            ).split(",")
            if f.strip()
        }
        self.exclude_patterns = tuple(
            pattern.strip()
            for pattern in os.environ.get(
                "MEOW_LGB_EXCLUDE_PATTERNS",
                "midpx_level_rank_cs,lastpx_level_rank_cs,high_level_rank_cs,"
                "low_level_rank_cs,open_level_rank_cs,bid0_level_rank_cs,ask0_level_rank_cs",
            ).split(",")
            if pattern.strip()
        )
        self._X_reservoir = None
        self._y_reservoir = None
        self._grp_reservoir = None  # group keys for ranking
        self._n_accumulated = 0
        self._model = None
        self._feature_names = None
        self._group_counter = 0
        self._group_map = {}  # (date, interval) -> group_id
        self._rng = np.random.RandomState(42)

    def reset(self):
        self._X_reservoir = None
        self._y_reservoir = None
        self._grp_reservoir = None
        self._n_accumulated = 0
        self._model = None
        self._feature_names = None
        self._group_counter = 0
        self._group_map = {}

    def partial_fit(self, xdf, ydf):
        if self._feature_names is None:
            cols = [c for c in xdf.columns if self._keep_column(c)]
            self._feature_names = cols
        x = xdf[self._feature_names].to_numpy(dtype=np.float32)
        y = ydf.to_numpy(dtype=np.float32).ravel()
        grp = self._build_group_keys(xdf) if self.is_ranking else None

        n = len(x)
        if self._X_reservoir is None:
            take = min(n, self.max_rows)
            self._X_reservoir = x[:take].copy()
            self._y_reservoir = y[:take].copy()
            if grp is not None:
                self._grp_reservoir = grp[:take].copy()
            self._n_accumulated = n
            return

        capacity = self._X_reservoir.shape[0]
        if capacity < self.max_rows:
            take = min(n, self.max_rows - capacity)
            self._X_reservoir = np.concatenate([self._X_reservoir, x[:take]], axis=0)
            self._y_reservoir = np.concatenate([self._y_reservoir, y[:take]], axis=0)
            if grp is not None:
                self._grp_reservoir = np.concatenate(
                    [self._grp_reservoir, grp[:take]], axis=0
                )
        else:
            for i in range(n):
                j = self._rng.randint(0, self._n_accumulated + i + 1)
                if j < capacity:
                    self._X_reservoir[j] = x[i]
                    self._y_reservoir[j] = y[i]
                    if grp is not None:
                        self._grp_reservoir[j] = grp[i]
        self._n_accumulated += n

    def _build_group_keys(self, xdf):
        """Assign integer group IDs for (date, interval) pairs."""
        idx_frame = xdf.index.to_frame(index=False)
        keys = []
        for _, row in idx_frame.iterrows():
            key = (int(row["date"]), int(row["interval"]))
            if key not in self._group_map:
                self._group_map[key] = self._group_counter
                self._group_counter += 1
            keys.append(self._group_map[key])
        return np.array(keys, dtype=np.int32)

    def finalize_fit(self):
        if self._X_reservoir is None or len(self._y_reservoir) < 1000:
            return
        params = dict(
            boosting_type="rf" if self.extra_trees else "gbdt",
            objective=self.objective,
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
        x_train = self._X_reservoir
        y_train = self._y_reservoir
        dataset_kwargs = dict(label=y_train, free_raw_data=False)
        if self.is_ranking and self._grp_reservoir is not None:
            sort_idx = np.argsort(self._grp_reservoir, kind="mergesort")
            x_train = x_train[sort_idx]
            y_train = y_train[sort_idx]
            ordered_grp = self._grp_reservoir[sort_idx]
            _, group_sizes = np.unique(ordered_grp, return_counts=True)
            dataset_kwargs["group"] = group_sizes.tolist()
            if self.objective == "lambdarank":
                params["lambdarank_truncation_level"] = self.lambdarank_truncation
            dataset_kwargs["label"] = y_train
        train_data = lgb.Dataset(x_train, **dataset_kwargs)
        self._model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
        )

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
