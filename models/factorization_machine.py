from __future__ import annotations

import os

import numpy as np


class ResidualFactorizationMachine:
    def __init__(self, input_dim: int):
        self.rank = max(int(os.environ.get("MEOW_FM_RANK", "8")), 1)
        self.batch_size = max(int(os.environ.get("MEOW_FM_BATCH_SIZE", "8192")), 256)
        self.lr = float(os.environ.get("MEOW_FM_LR", "0.01"))
        self.linear_l2 = float(os.environ.get("MEOW_FM_LINEAR_L2", "0.0005"))
        self.factor_l2 = float(os.environ.get("MEOW_FM_FACTOR_L2", "0.001"))
        self.bias_l2 = float(os.environ.get("MEOW_FM_BIAS_L2", "0.0"))
        self.grad_clip = float(os.environ.get("MEOW_FM_GRAD_CLIP", "0.1"))
        seed = int(os.environ.get("MEOW_FM_SEED", "20260529"))
        rng = np.random.default_rng(seed)

        self.bias = np.float32(0.0)
        self.linear = np.zeros(input_dim, dtype=np.float32)
        self.factors = rng.normal(0.0, 0.01, size=(input_dim, self.rank)).astype(np.float32)

        self.bias_accum = np.float32(1e-6)
        self.linear_accum = np.full(input_dim, 1e-6, dtype=np.float32)
        self.factor_accum = np.full((input_dim, self.rank), 1e-6, dtype=np.float32)

    def fit_chunk(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(x) == 0:
            return
        n_rows = x.shape[0]
        for start in range(0, n_rows, self.batch_size):
            stop = min(start + self.batch_size, n_rows)
            xb = np.ascontiguousarray(x[start:stop], dtype=np.float32)
            yb = np.ascontiguousarray(y[start:stop], dtype=np.float32)
            grad_bias, grad_linear, grad_factors = self._batch_gradients(xb, yb)
            self._adagrad_step(grad_bias, grad_linear, grad_factors)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return np.zeros(0, dtype=np.float32)
        x = np.ascontiguousarray(x, dtype=np.float32)
        linear_term = x @ self.linear
        xv = x @ self.factors
        x2v2 = np.square(x) @ np.square(self.factors)
        interaction_term = 0.5 * np.sum(np.square(xv) - x2v2, axis=1)
        return (self.bias + linear_term + interaction_term).astype(np.float32, copy=False)

    def _batch_gradients(self, x: np.ndarray, y: np.ndarray):
        pred = self.predict(x)
        err = pred - y
        batch_scale = np.float32(1.0 / max(len(x), 1))

        grad_bias = np.float32(err.mean() + self.bias_l2 * self.bias)
        grad_linear = (x.T @ err) * batch_scale + self.linear_l2 * self.linear

        xv = x @ self.factors
        x2 = np.square(x)
        grad_factors = (x.T @ (err[:, None] * xv)) * batch_scale
        grad_factors -= ((x2.T @ err)[:, None] * self.factors) * batch_scale
        grad_factors += self.factor_l2 * self.factors

        if self.grad_clip > 0.0:
            grad_bias = np.clip(grad_bias, -self.grad_clip, self.grad_clip)
            np.clip(grad_linear, -self.grad_clip, self.grad_clip, out=grad_linear)
            np.clip(grad_factors, -self.grad_clip, self.grad_clip, out=grad_factors)
        return grad_bias, grad_linear.astype(np.float32, copy=False), grad_factors.astype(np.float32, copy=False)

    def _adagrad_step(self, grad_bias, grad_linear, grad_factors) -> None:
        self.bias_accum = np.float32(self.bias_accum + grad_bias * grad_bias)
        self.linear_accum += np.square(grad_linear)
        self.factor_accum += np.square(grad_factors)

        self.bias = np.float32(self.bias - self.lr * grad_bias / np.sqrt(self.bias_accum))
        self.linear -= self.lr * grad_linear / np.sqrt(self.linear_accum)
        self.factors -= self.lr * grad_factors / np.sqrt(self.factor_accum)
