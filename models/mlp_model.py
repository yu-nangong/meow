"""sklearn MLPRegressor base model with reservoir-sampled chunked training.

Neural network architecture for learning nonlinear interactions between
symz (temporal) and rank_cs (cross-sectional) feature dimensions that
Ridge+LGB linear/tree architectures may miss.
"""
from __future__ import annotations

import os

import numpy as np


class MLPModel:
    def __init__(self):
        self.max_rows = int(os.environ.get("MEOW_MLP_MAX_ROWS", "500000"))
        self.hidden_1 = int(os.environ.get("MEOW_MLP_HIDDEN1", "128"))
        self.hidden_2 = int(os.environ.get("MEOW_MLP_HIDDEN2", "64"))
        self.learning_rate_init = float(os.environ.get("MEOW_MLP_LR", "0.001"))
        self.alpha = float(os.environ.get("MEOW_MLP_ALPHA", "0.0001"))
        self.batch_size = int(os.environ.get("MEOW_MLP_BATCH_SIZE", "256"))
        self.max_epochs = int(os.environ.get("MEOW_MLP_MAX_EPOCHS", "100"))
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
        # Select columns — MLP sees ALL features (no family exclusion)
        if self._feature_names is None:
            self._feature_names = list(xdf.columns)
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
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        # Standardize features for neural network stability
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self._X_reservoir)
        y_scaled = self._y_reservoir.copy()

        hidden = (self.hidden_1, self.hidden_2) if self.hidden_2 > 0 else (self.hidden_1,)
        self._model = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=self.alpha,
            batch_size=min(self.batch_size, len(X_scaled)),
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_epochs,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
            verbose=False,
        )
        self._model.fit(X_scaled, y_scaled)
        self._scaler = scaler

    def predict(self, xdf):
        if self._model is None or self._feature_names is None:
            return np.zeros(len(xdf), dtype=np.float64)
        x = xdf[self._feature_names].to_numpy(dtype=np.float32)
        x_scaled = self._scaler.transform(x)
        return self._model.predict(x_scaled).astype(np.float64)
