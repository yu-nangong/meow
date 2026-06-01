"""
Fast Gradient Boosting base model — fundamentally nonlinear replacement for Ridge.

Uses LightGBM (GBDT mode) with simple accumulation (no reservoir sampling).
Each chunk contributes up to max_rows via concatenation; training uses the
full accumulated buffer. GBRT captures nonlinear feature interactions and
threshold effects that linear models miss.

Environment variables:
  MEOW_GBRT_MAX_ROWS       — training buffer capacity (default 600000)
  MEOW_GBRT_NUM_LEAVES     — tree leaves (default 63)
  MEOW_GBRT_N_ESTIMATORS   — boosting rounds (default 300)
  MEOW_GBRT_LR             — learning rate (default 0.03)
  MEOW_GBRT_SUBSAMPLE      — row subsample per tree (default 0.7)
  MEOW_GBRT_COLSAMPLE      — feature subsample per tree (default 0.7)
  MEOW_GBRT_MIN_DATA       — min data in leaf (default 50)
  MEOW_GBRT_REG_LAMBDA     — L2 regularization (default 0.5)
  MEOW_GBRT_BAGGING_FREQ   — bagging frequency (default 5)
  MEOW_GBRT_BAGGING_FRAC   — bagging fraction (default 0.8)
"""

from __future__ import annotations

import os
import numpy as np
import lightgbm as lgb


class GBRTModel:
    def __init__(self, cacheDir: str | None = None):
        self.max_rows = int(os.environ.get("MEOW_GBRT_MAX_ROWS", "600000"))
        self.num_leaves = int(os.environ.get("MEOW_GBRT_NUM_LEAVES", "63"))
        self.n_estimators = int(os.environ.get("MEOW_GBRT_N_ESTIMATORS", "300"))
        self.learning_rate = float(os.environ.get("MEOW_GBRT_LR", "0.03"))
        self.subsample = float(os.environ.get("MEOW_GBRT_SUBSAMPLE", "0.7"))
        self.colsample = float(os.environ.get("MEOW_GBRT_COLSAMPLE", "0.7"))
        self.min_data = int(os.environ.get("MEOW_GBRT_MIN_DATA", "50"))
        self.reg_lambda = float(os.environ.get("MEOW_GBRT_REG_LAMBDA", "0.5"))
        self.bagging_freq = int(os.environ.get("MEOW_GBRT_BAGGING_FREQ", "5"))
        self.bagging_frac = float(os.environ.get("MEOW_GBRT_BAGGING_FRAC", "0.8"))

        self._X_buf = None
        self._y_buf = None
        self._n_accumulated = 0
        self._model = None
        self._feature_names = None

    def reset(self):
        self._X_buf = None
        self._y_buf = None
        self._n_accumulated = 0
        self._model = None
        self._feature_names = None

    def partial_fit(self, xdf, ydf):
        if self._feature_names is None:
            self._feature_names = list(xdf.columns)
        x = xdf[self._feature_names].to_numpy(dtype=np.float32)
        y = ydf.to_numpy(dtype=np.float32).ravel()

        n = len(x)
        if self._X_buf is None:
            take = min(n, self.max_rows)
            self._X_buf = x[:take].copy()
            self._y_buf = y[:take].copy()
            self._n_accumulated = take
        elif self._n_accumulated < self.max_rows:
            take = min(n, self.max_rows - self._n_accumulated)
            self._X_buf = np.concatenate([self._X_buf, x[:take]], axis=0)
            self._y_buf = np.concatenate([self._y_buf, y[:take]], axis=0)
            self._n_accumulated += take

    def finalize_fit(self):
        if self._X_buf is None or len(self._y_buf) < 1000:
            return
        params = dict(
            boosting_type="gbdt",
            objective="regression",
            metric="l2",
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            subsample_freq=self.bagging_freq,
            colsample_bytree=self.colsample,
            min_child_samples=self.min_data,
            reg_lambda=self.reg_lambda,
            bagging_fraction=self.bagging_frac,
            bagging_freq=self.bagging_freq,
            verbose=-1,
            random_state=42,
            n_jobs=1,
            force_col_wise=True,  # faster for many columns
        )
        train_data = lgb.Dataset(self._X_buf, label=self._y_buf, free_raw_data=False)
        self._model = lgb.train(params, train_data, num_boost_round=self.n_estimators)

    def predict(self, xdf):
        if self._model is None or self._feature_names is None:
            return np.zeros(len(xdf), dtype=np.float64)
        x = xdf[self._feature_names].to_numpy(dtype=np.float32)
        return self._model.predict(x).astype(np.float64)
