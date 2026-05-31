"""
Tiny numpy MLP — 2 hidden layers, ReLU, SGD with momentum.

Trains incrementally via partial_fit on each chunk (no accumulation).
Uses He initialization + L2 regularization + momentum SGD.
Memory: O(n_features * hidden) — independent of n_samples.

Reference: Hornik et al. (1989), "Multilayer feedforward networks are
universal approximators." Neural Networks, 2(5), 359-366.
"""
from __future__ import annotations

import os
import numpy as np


class TinyMLP:
    def __init__(self, cacheDir: str | None = None):
        self.hidden1 = int(os.environ.get("MEOW_MLP_HIDDEN1", "32"))
        self.hidden2 = int(os.environ.get("MEOW_MLP_HIDDEN2", "16"))
        self.lr = float(os.environ.get("MEOW_MLP_LR", "0.001"))
        self.momentum = float(os.environ.get("MEOW_MLP_MOMENTUM", "0.9"))
        self.l2 = float(os.environ.get("MEOW_MLP_L2", "0.001"))
        self.epochs_per_chunk = int(os.environ.get("MEOW_MLP_EPOCHS_PER_CHUNK", "3"))
        self.batch_size = int(os.environ.get("MEOW_MLP_BATCH_SIZE", "1024"))
        self._fitted = False
        self._n_features = None

    def reset(self):
        self._fitted = False
        self._n_features = None

    def partial_fit(self, xdf, ydf):
        x = xdf.to_numpy(dtype=np.float64, copy=False)
        y = ydf.to_numpy(dtype=np.float64, copy=False).ravel()

        if not self._fitted:
            self._init_params(x.shape[1])
            self._x_mean = x.mean(axis=0)
            self._x_std = x.std(axis=0).clip(min=1e-12)
            self._y_mean = y.mean()
            self._y_std = y.std()
            self._fitted = True

        X_norm = (x - self._x_mean) / self._x_std
        y_norm = (y - self._y_mean) / self._y_std.clip(min=1e-12)

        n = len(x)
        rng = np.random.RandomState(42)
        for epoch in range(self.epochs_per_chunk):
            idx = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch = idx[start:start + self.batch_size]
                xb = X_norm[batch]
                yb = y_norm[batch, None]

                z1 = xb @ self._w1 + self._b1
                a1 = np.maximum(z1, 0.0)
                z2 = a1 @ self._w2 + self._b2
                a2 = np.maximum(z2, 0.0)
                y_pred = a2 @ self._w3 + self._b3

                dy = 2.0 * (y_pred - yb) / len(batch)

                dw3 = a2.T @ dy
                db3 = dy.sum(axis=0)
                da2 = dy @ self._w3.T
                dz2 = da2 * (z2 > 0.0)
                dw2 = a1.T @ dz2
                db2 = dz2.sum(axis=0)
                da1 = dz2 @ self._w2.T
                dz1 = da1 * (z1 > 0.0)
                dw1 = xb.T @ dz1
                db1 = dz1.sum(axis=0)

                dw1 += self.l2 * self._w1
                dw2 += self.l2 * self._w2
                dw3 += self.l2 * self._w3

                self._vw1 = self.momentum * self._vw1 + self.lr * dw1
                self._vb1 = self.momentum * self._vb1 + self.lr * db1
                self._vw2 = self.momentum * self._vw2 + self.lr * dw2
                self._vb2 = self.momentum * self._vb2 + self.lr * db2
                self._vw3 = self.momentum * self._vw3 + self.lr * dw3
                self._vb3 = self.momentum * self._vb3 + self.lr * db3

                self._w1 -= self._vw1
                self._b1 -= self._vb1
                self._w2 -= self._vw2
                self._b2 -= self._vb2
                self._w3 -= self._vw3
                self._b3 -= self._vb3

    def finalize_fit(self):
        pass

    def fit(self, xdf, ydf):
        self.reset()
        self.partial_fit(xdf, ydf)
        self.finalize_fit()

    def _init_params(self, n_features):
        rng = np.random.RandomState(42)
        scale1 = np.sqrt(2.0 / n_features)
        scale2 = np.sqrt(2.0 / self.hidden1)
        scale3 = np.sqrt(2.0 / self.hidden2)

        self._w1 = rng.randn(n_features, self.hidden1).astype(np.float64) * scale1
        self._b1 = np.zeros(self.hidden1, dtype=np.float64)
        self._w2 = rng.randn(self.hidden1, self.hidden2).astype(np.float64) * scale2
        self._b2 = np.zeros(self.hidden2, dtype=np.float64)
        self._w3 = rng.randn(self.hidden2, 1).astype(np.float64) * scale3
        self._b3 = np.zeros(1, dtype=np.float64)

        self._vw1 = np.zeros_like(self._w1)
        self._vb1 = np.zeros_like(self._b1)
        self._vw2 = np.zeros_like(self._w2)
        self._vb2 = np.zeros_like(self._b2)
        self._vw3 = np.zeros_like(self._w3)
        self._vb3 = np.zeros_like(self._b3)

        self._n_features = n_features

    def predict(self, xdf):
        if not self._fitted:
            return np.zeros(len(xdf), dtype=np.float64)
        x = xdf.to_numpy(dtype=np.float64, copy=False)
        X_norm = (x - self._x_mean) / self._x_std
        z1 = X_norm @ self._w1 + self._b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self._w2 + self._b2
        a2 = np.maximum(z2, 0.0)
        y_pred_norm = a2 @ self._w3 + self._b3
        return (y_pred_norm.ravel() * self._y_std) + self._y_mean

    def fit_from_arrays(self, X, y):
        """Fit from pre-accumulated arrays (used by solution.py for MLP)."""
        self.reset()
        self._init_params(X.shape[1])
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0).clip(min=1e-12)
        self._y_mean = y.mean()
        self._y_std = y.std()

        X_norm = (X - self._x_mean) / self._x_std
        y_norm = (y - self._y_mean) / self._y_std.clip(min=1e-12)
        self._fitted = True

        n = len(X)
        rng = np.random.RandomState(42)
        from log import log
        for epoch in range(self.epochs_per_chunk * 3):
            idx = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch = idx[start:start + self.batch_size]
                xb = X_norm[batch]
                yb = y_norm[batch, None]

                z1 = xb @ self._w1 + self._b1
                a1 = np.maximum(z1, 0.0)
                z2 = a1 @ self._w2 + self._b2
                a2 = np.maximum(z2, 0.0)
                y_pred = a2 @ self._w3 + self._b3

                dy = 2.0 * (y_pred - yb) / len(batch)

                dw3 = a2.T @ dy
                db3 = dy.sum(axis=0)
                da2 = dy @ self._w3.T
                dz2 = da2 * (z2 > 0.0)
                dw2 = a1.T @ dz2
                db2 = dz2.sum(axis=0)
                da1 = dz2 @ self._w2.T
                dz1 = da1 * (z1 > 0.0)
                dw1 = xb.T @ dz1
                db1 = dz1.sum(axis=0)

                dw1 += self.l2 * self._w1
                dw2 += self.l2 * self._w2
                dw3 += self.l2 * self._w3

                self._vw1 = self.momentum * self._vw1 + self.lr * dw1
                self._vb1 = self.momentum * self._vb1 + self.lr * db1
                self._vw2 = self.momentum * self._vw2 + self.lr * dw2
                self._vb2 = self.momentum * self._vb2 + self.lr * db2
                self._vw3 = self.momentum * self._vw3 + self.lr * dw3
                self._vb3 = self.momentum * self._vb3 + self.lr * db3

                self._w1 -= self._vw1
                self._b1 -= self._vb1
                self._w2 -= self._vw2
                self._b2 -= self._vb2
                self._w3 -= self._vw3
                self._b3 -= self._vb3
            train_loss = float(np.mean((X_norm @ self._w1 + self._b1 @ 0) ** 2))
            if epoch == 0 or (epoch + 1) % 5 == 0:
                log.inf(f"  MLP epoch {epoch+1}: loss check")

        log.inf(f"MLP fitted: {X.shape[1]}->{self.hidden1}->{self.hidden2}->1")
