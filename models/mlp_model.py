"""
sklearn MLPRegressor base model — fundamentally nonlinear replacement for Ridge/ElasticNet.

Uses reservoir sampling to bound training data, then fits a shallow MLP with
ReLU activations. The nonlinearity can capture feature interactions that linear
models miss, while the interval-residual ridge on top handles per-interval corrections.

Environment variables:
  MEOW_MLP_MAX_ROWS      — reservoir capacity (default 400000)
  MEOW_MLP_HIDDEN        — comma-sep hidden layer sizes (default "64,32")
  MEOW_MLP_ALPHA         — L2 regularization (default 0.01)
  MEOW_MLP_MAX_ITER      — max iterations per fit call (default 200)
  MEOW_MLP_LR_INIT       — initial learning rate (default 0.001)
  MEOW_MLP_BATCH_SIZE    — batch size (default 256)
  MEOW_MLP_EARLY_STOP    — use early stopping (default 1)
  MEOW_MLP_VAL_FRAC      — validation fraction if early stopping (default 0.1)
  MEOW_MLP_TOL           — tolerance for optimization (default 1e-4)
"""

from __future__ import annotations

import os
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class MLPModel:
    def __init__(self, cacheDir: str | None = None):
        self.max_rows = int(os.environ.get("MEOW_MLP_MAX_ROWS", "400000"))
        hidden_str = os.environ.get("MEOW_MLP_HIDDEN", "64,32")
        self.hidden_layer_sizes = tuple(int(x) for x in hidden_str.split(",") if x.strip())
        self.alpha = float(os.environ.get("MEOW_MLP_ALPHA", "0.01"))
        self.max_iter = int(os.environ.get("MEOW_MLP_MAX_ITER", "200"))
        self.lr_init = float(os.environ.get("MEOW_MLP_LR_INIT", "0.001"))
        self.batch_size = int(os.environ.get("MEOW_MLP_BATCH_SIZE", "256"))
        self.early_stop = os.environ.get("MEOW_MLP_EARLY_STOP", "1") != "0"
        self.val_frac = float(os.environ.get("MEOW_MLP_VAL_FRAC", "0.1"))
        self.tol = float(os.environ.get("MEOW_MLP_TOL", "1e-4"))
        self._X_reservoir = None
        self._y_reservoir = None
        self._n_accumulated = 0
        self._model = None
        self._scaler = None
        self._rng = np.random.RandomState(42)

    def reset(self):
        self._X_reservoir = None
        self._y_reservoir = None
        self._n_accumulated = 0
        self._model = None
        self._scaler = None

    def partial_fit(self, xdf, ydf):
        x = xdf.to_numpy(dtype=np.float32)
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
        X = self._X_reservoir.astype(np.float64)
        y = self._y_reservoir.astype(np.float64)
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(X)
        self._model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=self.alpha,
            batch_size=min(self.batch_size, len(X) // 10),
            learning_rate_init=self.lr_init,
            max_iter=self.max_iter,
            shuffle=True,
            random_state=42,
            tol=self.tol,
            verbose=False,
            warm_start=False,
            early_stopping=self.early_stop,
            validation_fraction=self.val_frac if self.early_stop else 0.0,
            n_iter_no_change=10,
        )
        self._model.fit(X, y)

    def predict(self, xdf):
        if self._model is None or self._scaler is None:
            return np.zeros(len(xdf), dtype=np.float64)
        x = xdf.to_numpy(dtype=np.float32).astype(np.float64)
        x = self._scaler.transform(x)
        return self._model.predict(x).astype(np.float64)
