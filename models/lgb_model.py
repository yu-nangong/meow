"""LightGBM base model with reservoir-sampled chunked training.

Supports custom Pearson correlation objective when MEOW_LGB_PEARSON_OBJ=1,
directly optimizing for the evaluation metric.
"""
from __future__ import annotations

import os

import numpy as np
import lightgbm as lgb


def _pearson_grad_hess(preds, train_data):
    """Custom objective: negative Pearson correlation (to minimize).

    Falls back to MSE when predictions are degenerate (constant).
    """
    y = train_data.get_label()
    n = len(y)
    y_std = y.std()
    p_std = preds.std()
    eps = 1e-8
    if y_std < eps or p_std < eps:
        grad = 2.0 * (preds - y) / n
        hess = np.full(n, 2.0 / n, dtype=np.float64)
        return grad, hess
    y_mean = y.mean()
    p_mean = preds.mean()
    r = np.corrcoef(y, preds)[0, 1]
    grad = -(y - y_mean) / (n * y_std * p_std) + r * (preds - p_mean) / (n * p_std * p_std)
    hess = np.full(n, 1.0 / (n * p_std * p_std), dtype=np.float64)
    return grad, hess


def _pearson_eval(preds, train_data):
    """Feval: Pearson correlation (higher is better)."""
    y = train_data.get_label()
    r = np.corrcoef(y, preds)[0, 1]
    return "pearson", r, True


class LGBModel:
    def __init__(self):
        self.max_rows = int(os.environ.get("MEOW_LGB_MAX_ROWS", "800000"))
        self.num_leaves = int(os.environ.get("MEOW_LGB_NUM_LEAVES", "31"))
        self.learning_rate = float(os.environ.get("MEOW_LGB_LEARNING_RATE", "0.05"))
        self.n_estimators = int(os.environ.get("MEOW_LGB_N_ESTIMATORS", "200"))
        self.extra_trees = os.environ.get("MEOW_LGB_EXTRA_TREES", "0") != "0"
        self.subsample = float(os.environ.get("MEOW_LGB_SUBSAMPLE", "0.8"))
        self.colsample_bytree = float(os.environ.get("MEOW_LGB_COLSAMPLE_BYTREE", "0.8"))
        self.min_child_samples = int(os.environ.get("MEOW_LGB_MIN_CHILD_SAMPLES", "100"))
        self.reg_lambda = float(os.environ.get("MEOW_LGB_REG_LAMBDA", "1.0"))
        self.pearson_obj = os.environ.get("MEOW_LGB_PEARSON_OBJ", "0") != "0"
        self.exclude_families = {
            f.strip()
            for f in os.environ.get(
                "MEOW_LGB_EXCLUDE_FAMILIES",
                "cs,time_interaction,u_interaction,time_sq_interaction,u_sq_interaction",
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
        self._n_accumulated = 0
        self._model = None
        self._feature_names = None
        self._rng = np.random.RandomState(42)

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
        y = ydf.to_numpy(dtype=np.float32).ravel()

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
        params = dict(
            boosting_type="rf" if self.extra_trees else "gbdt",
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
            self._X_reservoir, label=self._y_reservoir, free_raw_data=False,
        )
        if self.pearson_obj:
            params["objective"] = _pearson_grad_hess
            params["boost_from_average"] = True
        self._model = lgb.train(
            params, train_data, num_boost_round=self.n_estimators,
            feval=_pearson_eval if self.pearson_obj else None,
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
