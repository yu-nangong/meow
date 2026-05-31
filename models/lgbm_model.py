"""
LightGBM base model for MEOW — replaces Ridge/ElasticNet with gradient boosting.

LightGBM captures non-linear feature interactions and threshold effects that
linear models miss. On tabular financial data with ~190 features, gradient
boosting typically outperforms penalized linear regression.

Uses chunked accumulation with subsampling to stay within grader memory,
then a single final fit on the pooled subsample.

Reference: Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (2017)
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import lightgbm as lgb


class LightGBMModel:
    def __init__(self, cacheDir: str | None = None):
        self.n_estimators = int(os.environ.get("MEOW_LGBM_N_ESTIMATORS", "200"))
        self.num_leaves = int(os.environ.get("MEOW_LGBM_NUM_LEAVES", "15"))
        self.learning_rate = float(os.environ.get("MEOW_LGBM_LEARNING_RATE", "0.1"))
        self.subsample = float(os.environ.get("MEOW_LGBM_SUBSAMPLE", "0.7"))
        self.colsample_bytree = float(os.environ.get("MEOW_LGBM_COLSAMPLE_BY_TREE", "0.7"))
        self.min_child_samples = int(os.environ.get("MEOW_LGBM_MIN_CHILD_SAMPLES", "50"))
        self.reg_alpha = float(os.environ.get("MEOW_LGBM_REG_ALPHA", "0.1"))
        self.reg_lambda = float(os.environ.get("MEOW_LGBM_REG_LAMBDA", "1.0"))
        self.max_bin = int(os.environ.get("MEOW_LGBM_MAX_BIN", "255"))
        self.verbose = int(os.environ.get("MEOW_LGBM_VERBOSE", "-1"))
        self.num_threads = int(os.environ.get("MEOW_LGBM_NUM_THREADS", "2"))
        # Subsampling for memory/time control
        self.max_train_samples = int(os.environ.get("MEOW_LGBM_MAX_TRAIN_SAMPLES", "300000"))
        self._X_list: list[np.ndarray] = []
        self._y_list: list[np.ndarray] = []
        self._feature_names: list[str] = []
        self._model: lgb.Booster | None = None
        self._fitted = False

    def reset(self):
        self._X_list = []
        self._y_list = []
        self._feature_names = []
        self._model = None
        self._fitted = False

    def partial_fit(self, xdf, ydf):
        x = xdf.to_numpy(dtype=np.float32, copy=False)
        y = ydf.to_numpy(dtype=np.float32, copy=False).ravel()
        self._X_list.append(x)
        self._y_list.append(y)
        if not self._feature_names:
            self._feature_names = list(xdf.columns)

    def finalize_fit(self):
        if not self._X_list:
            return
        X = np.concatenate(self._X_list, axis=0).astype(np.float64)
        y = np.concatenate(self._y_list, axis=0).astype(np.float64)
        self._X_list = []
        self._y_list = []

        n = len(X)
        if n > self.max_train_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, self.max_train_samples, replace=False)
            X, y = X[idx], y[idx]

        train_data = lgb.Dataset(X, y, feature_name=self._feature_names, free_raw_data=False)

        params = {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "subsample_freq": 1,
            "colsample_bytree": self.colsample_bytree,
            "min_child_samples": self.min_child_samples,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "max_bin": self.max_bin,
            "verbose": self.verbose,
            "num_threads": self.num_threads,
            "seed": 42,
            "deterministic": True,
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = lgb.train(
                params,
                train_data,
                num_boost_round=self.n_estimators,
                valid_sets=None,
            )

        self._fitted = True
        from log import log
        n_features_used = self._model.num_feature()
        log.inf(f"LightGBM fitted: {self.n_estimators} trees, {self.num_leaves} leaves, {n_features_used} features")

    def fit(self, xdf, ydf):
        self.reset()
        self.partial_fit(xdf, ydf)
        self.finalize_fit()

    def predict(self, xdf):
        if not self._fitted or self._model is None:
            return np.zeros(len(xdf), dtype=np.float64)
        x = xdf.to_numpy(dtype=np.float64, copy=False)
        return self._model.predict(x).astype(np.float64)
