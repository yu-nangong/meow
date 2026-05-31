"""
Lightweight online NN — single hidden layer, ReLU, trained via SGD.

Key design choices for speed & memory:
- Single hidden layer (D->h->1) — minimal params
- SGDR (Stochastic Gradient Descent with Restarts) — 1 pass only
- Online updates per chunk — no data accumulation
- Float32 weights to save memory
- Batch normalization via running stats

This REPLACES the 2-stage (base model + interval residual) entirely.
"""
from __future__ import annotations

import os
import math
import numpy as np


class OnlineNN:
    def __init__(self, cacheDir: str | None = None):
        self.hidden = int(os.environ.get("MEOW_NN_HIDDEN", "64"))
        self.lr = float(os.environ.get("MEOW_NN_LR", "0.005"))
        self.momentum = float(os.environ.get("MEOW_NN_MOMENTUM", "0.9"))
        self.l2 = float(os.environ.get("MEOW_NN_L2", "1e-5"))
        self.batch_size = int(os.environ.get("MEOW_NN_BATCH_SIZE", "2048"))
        self.n_epochs = int(os.environ.get("MEOW_NN_EPOCHS", "1"))
        self._fitted = False
        self._n_features = None

    def reset(self):
        self._fitted = False
        self._n_features = None

    def partial_fit(self, xdf, ydf):
        x = xdf.to_numpy(dtype=np.float32, copy=False)
        y = ydf.to_numpy(dtype=np.float32, copy=False).ravel()

        if not self._fitted:
            self._init_params(x.shape[1])
            self._n_features = x.shape[1]
            # Running stats for input normalization
            self._running_mean = np.zeros(x.shape[1], dtype=np.float32)
            self._running_var = np.ones(x.shape[1], dtype=np.float32)
            self._running_n = 0
            self._y_mean = 0.0
            self._y_std = 1.0
            self._fitted = True

        # Update running stats
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        n = len(x)
        old_n = self._running_n

        if old_n == 0:
            self._running_mean = batch_mean.astype(np.float32)
            self._running_var = batch_var.astype(np.float32)
            self._running_n = n
        else:
            new_mean = (old_n * self._running_mean + n * batch_mean) / (old_n + n)
            delta = batch_mean - self._running_mean
            m_a = self._running_var * old_n
            m_b = batch_var * n
            M2 = m_a + m_b + delta**2 * old_n * n / (old_n + n)
            self._running_mean = new_mean.astype(np.float32)
            self._running_var = (M2 / (old_n + n)).astype(np.float32)
            self._running_n = old_n + n

        # Normalize inputs
        scale = np.sqrt(np.maximum(self._running_var, 1e-8)).astype(np.float32)
        X_norm = (x - self._running_mean) / scale

        # Normalize target (running)
        y_batch_mean = y.mean()
        y_batch_std = max(y.std(), 1e-8)
        if self._running_n == n:  # first batch
            self._y_mean = y_batch_mean
            self._y_std = y_batch_std
        else:
            alpha = min(n / max(self._running_n, 1), 0.1)
            self._y_mean = (1 - alpha) * self._y_mean + alpha * y_batch_mean
            self._y_std = max((1 - alpha) * self._y_std + alpha * y_batch_std, 1e-8)

        y_norm = (y - self._y_mean) / self._y_std

        # SGD training
        rng = np.random.RandomState(42 + self._n_chunks_so_far())
        for _ in range(self.n_epochs):
            idx = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch = idx[start:start + self.batch_size]
                xb = X_norm[batch]
                yb = y_norm[batch]

                # Forward
                z1 = xb @ self._w1 + self._b1
                a1 = np.maximum(z1, 0.0)  # ReLU
                pred = a1 @ self._w2 + self._b2

                # MSE loss gradient
                dy = 2.0 * (pred - yb[:, None]) / len(batch)

                # Backward
                dw2 = a1.T @ dy
                db2 = dy.sum(axis=0)
                da1 = dy @ self._w2.T
                dz1 = da1 * (z1 > 0).astype(np.float32)
                dw1 = xb.T @ dz1
                db1 = dz1.sum(axis=0)

                # L2 regularization
                dw1 += self.l2 * self._w1
                dw2 += self.l2 * self._w2

                # Momentum SGD update
                self._vw1 = self.momentum * self._vw1 - self.lr * dw1
                self._vb1 = self.momentum * self._vb1 - self.lr * db1
                self._vw2 = self.momentum * self._vw2 - self.lr * dw2
                self._vb2 = self.momentum * self._vb2 - self.lr * db2

                self._w1 += self._vw1
                self._b1 += self._vb1
                self._w2 += self._vw2
                self._b2 += self._vb2

    def finalize_fit(self):
        pass

    def fit(self, xdf, ydf):
        self.reset()
        self.partial_fit(xdf, ydf)

    def predict(self, xdf):
        if not self._fitted:
            return np.zeros(len(xdf), dtype=np.float64)
        x = xdf.to_numpy(dtype=np.float32, copy=False)
        scale = np.sqrt(np.maximum(self._running_var, 1e-8)).astype(np.float32)
        X_norm = (x - self._running_mean) / scale

        z1 = X_norm @ self._w1 + self._b1
        a1 = np.maximum(z1, 0.0)
        pred = (a1 @ self._w2 + self._b2).ravel()
        return (pred * self._y_std + self._y_mean).astype(np.float64)

    def _init_params(self, n_features):
        # He initialization
        rng = np.random.RandomState(42)
        scale1 = math.sqrt(2.0 / n_features)
        self._w1 = (rng.randn(n_features, self.hidden) * scale1).astype(np.float32)
        self._b1 = np.zeros(self.hidden, dtype=np.float32)
        scale2 = math.sqrt(2.0 / self.hidden)
        self._w2 = (rng.randn(self.hidden, 1) * scale2).astype(np.float32)
        self._b2 = np.zeros(1, dtype=np.float32)
        self._vw1 = np.zeros_like(self._w1)
        self._vb1 = np.zeros_like(self._b1)
        self._vw2 = np.zeros_like(self._w2)
        self._vb2 = np.zeros_like(self._b2)

    def _n_chunks_so_far(self):
        if not hasattr(self, '_chunk_count'):
            self._chunk_count = 0
        self._chunk_count += 1
        return self._chunk_count
