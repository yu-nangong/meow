"""Pearson-optimized neural network: trains directly on Pearson correlation loss.

This is fundamentally different from all existing models (Ridge, LGB, ElasticNet)
which optimize MSE. Since the evaluation metric is Pearson correlation, directly
optimizing it may find solutions that MSE-optimal models cannot reach.

Architecture: 251 features -> 64 -> 32 -> 1 (ReLU activations)
Loss: 1 - Pearson(y_pred, y_true) over each batch
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn


def pearson_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Differentiable 1 - Pearson correlation loss over a batch."""
    y_pred = y_pred.squeeze(-1)
    y_true = y_true.squeeze(-1)

    pred_mean = y_pred.mean()
    true_mean = y_true.mean()
    pred_centered = y_pred - pred_mean
    true_centered = y_true - true_mean

    cov = (pred_centered * true_centered).mean()
    pred_std = torch.sqrt((pred_centered ** 2).mean() + 1e-8)
    true_std = torch.sqrt((true_centered ** 2).mean() + 1e-8)

    corr = cov / (pred_std * true_std)
    # Clamp correlation to [-1, 1] for numerical stability
    corr = torch.clamp(corr, -0.9999, 0.9999)
    return 1.0 - corr


class PearsonNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PearsonNNModel:
    """Scikit-learn-compatible interface for Pearson-optimized neural network."""

    def __init__(self):
        self.lr = float(os.environ.get("MEOW_PN_LR", "1e-3"))
        self.epochs = int(os.environ.get("MEOW_PN_EPOCHS", "3"))
        self.batch_size = int(os.environ.get("MEOW_PN_BATCH_SIZE", "16384"))
        self.hidden = int(os.environ.get("MEOW_PN_HIDDEN", "64"))
        self.weight_decay = float(os.environ.get("MEOW_PN_WEIGHT_DECAY", "1e-5"))
        self._model = None
        self._optimizer = None
        self._n_features = None
        self._feature_names = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def reset(self):
        self._model = None
        self._optimizer = None
        self._n_features = None
        self._feature_names = None

    def _init_model(self, n_features: int):
        self._n_features = n_features
        self._model = PearsonNet(n_features, self.hidden).to(self._device)
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def partial_fit(self, xdf, ydf):
        x = xdf.to_numpy(dtype=np.float32)
        y = ydf.to_numpy(dtype=np.float32).reshape(-1, 1)

        n = len(x)
        if self._model is None:
            self._n_features = x.shape[1]
            self._feature_names = list(xdf.columns)
            self._init_model(self._n_features)

        # Normalize features per chunk for stable training
        x_mean = np.nanmean(x, axis=0)
        x_std = np.nanstd(x, axis=0)
        x_std = np.where(x_std < 1e-8, 1.0, x_std)
        x = (x - x_mean) / x_std

        # Normalize target per chunk
        y_mean = np.nanmean(y)
        y_std = np.nanstd(y)
        y_std = y_std if y_std > 1e-8 else 1.0
        y = (y - y_mean) / y_std

        x_t = torch.from_numpy(x).to(self._device)
        y_t = torch.from_numpy(y).to(self._device)

        self._model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self._device)
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                xb, yb = x_t[idx], y_t[idx]

                self._optimizer.zero_grad()
                pred = self._model(xb)
                loss = pearson_loss(pred, yb)
                loss.backward()
                self._optimizer.step()

    def finalize_fit(self):
        pass  # Online-style training, no finalization needed

    def predict(self, xdf):
        if self._model is None:
            return np.zeros(len(xdf), dtype=np.float64)
        x = xdf.to_numpy(dtype=np.float32)
        # Use simple standardization; the model learns to handle scale
        x_mean = np.nanmean(x, axis=0)
        x_std = np.nanstd(x, axis=0)
        x_std = np.where(x_std < 1e-8, 1.0, x_std)
        x = (x - x_mean) / x_std

        self._model.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(x).to(self._device)
            pred = self._model(x_t).cpu().numpy().ravel()
        return pred.astype(np.float64)
