"""
Incremental LightGBM for MEOW — no concatenation, no two-stage residual.

Trains incrementally via init_model from each chunk. Keeps memory flat.
Directly predicts fret12 using all features including interval/time context.

Reference: Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (2017)
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import lightgbm as lgb


class LightGBMIncremental:
    def __init__(self, cacheDir: str | None = None):
        self.num_leaves = int(os.environ.get("MEOW_LGBM_NUM_LEAVES", "4"))
        self.learning_rate = float(os.environ.get("MEOW_LGBM_LEARNING_RATE", "0.03"))
        self.n_estimators = int(os.environ.get("MEOW_LGBM_N_ESTIMATORS", "10"))
        self.subsample = float(os.environ.get("MEOW_LGBM_SUBSAMPLE", "0.6"))
        self.colsample_bytree = float(os.environ.get("MEOW_LGBM_COLSAMPLE_BY_TREE", "0.5"))
        self.min_child_samples = int(os.environ.get("MEOW_LGBM_MIN_CHILD_SAMPLES", "100"))
        self.reg_alpha = float(os.environ.get("MEOW_LGBM_REG_ALPHA", "0.5"))
        self.reg_lambda = float(os.environ.get("MEOW_LGBM_REG_LAMBDA", "2.0"))
        self.verbose = int(os.environ.get("MEOW_LGBM_VERBOSE", "-1"))
        self.num_threads = int(os.environ.get("MEOW_LGBM_NUM_THREADS", "2"))
        # Stable params
        self.min_gain_to_split = float(os.environ.get("MEOW_LGBM_MIN_GAIN", "0.1"))
        self.bagging_freq = int(os.environ.get("MEOW_LGBM_BAGGING_FREQ", "1"))
        self.max_bin = int(os.environ.get("MEOW_LGBM_MAX_BIN", "63"))

        self._model: lgb.Booster | None = None
        self._feature_names: list[str] = []
        self._fitted = False

    def reset(self):
        self._model = None
        self._feature_names = []
        self._fitted = False

    def partial_fit(self, xdf, ydf):
        """Train on one chunk, continuing from previous model."""
        x = xdf.to_numpy(dtype=np.float32, copy=False)
        y = ydf.to_numpy(dtype=np.float32, copy=False).ravel()
        names = list(xdf.columns)

        if not self._feature_names:
            self._feature_names = names
        if len(y) == 0:
            return

        # Subsample if chunk is huge
        n = len(x)
        max_chunk = 150000
        if n > max_chunk:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, max_chunk, replace=False)
            x, y = x[idx], y[idx]

        train_data = lgb.Dataset(x, y, feature_name=names, free_raw_data=True)

        params = {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "subsample_freq": self.bagging_freq,
            "colsample_bytree": self.colsample_bytree,
            "min_child_samples": self.min_child_samples,
            "min_gain_to_split": self.min_gain_to_split,
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
            if self._model is not None:
                # Continue boosting from previous model
                self._model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=self.n_estimators,
                    init_model=self._model,
                    valid_sets=None,
                )
            else:
                self._model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=self.n_estimators,
                    valid_sets=None,
                )

        self._fitted = True

    def finalize_fit(self):
        """No-op: model is already incrementally trained."""
        if self._model is not None:
            from log import log
            log.inf(
                f"LightGBM incremental done: {self._model.num_trees()} trees, "
                f"{self.num_leaves} max leaves"
            )

    def fit(self, xdf, ydf):
        self.reset()
        self.partial_fit(xdf, ydf)
        self.finalize_fit()

    def predict(self, xdf):
        if not self._fitted or self._model is None:
            return np.zeros(len(xdf), dtype=np.float64)
        x = xdf.to_numpy(dtype=np.float64, copy=False)
        return self._model.predict(x).astype(np.float64)
