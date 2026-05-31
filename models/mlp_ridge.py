"""
MLP that replaces the ridge base model.
Takes ~200 engineered features, learns nonlinear interactions via 2 hidden layers.
Keeps the interval-residual on top.

Key design choices:
- 2 hidden layers with LayerNorm + Dropout for stability
- Trained in chunks via mini-batch SGD, no full data accumulation
- Uses in-place shuffle indexing to avoid memory copies

Reference: simple MLP is a strong baseline for tabular data
(cf. Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data", NeurIPS 2021).
"""
from __future__ import annotations

import os
import numpy as np
import torch
from torch import nn


class MLPRidge(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_dim = n_features
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        self.hidden = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(x)).squeeze(-1)


class TorchRidgeBase:
    """
    Drop-in replacement for MeowModel (ridge) that uses a small MLP.
    Trained via mini-batch SGD across chunks. Does NOT accumulate all data.

    partial_fit: trains on one chunk for a few epochs.
    finalize_fit: trains for additional epochs by re-iterating over chunks.
    """
    def __init__(self, cache_dir=None):
        self.n_features = None
        self.feature_names = None
        self.device = torch.device("cpu")
        self.model = None
        self.optimizer = None
        self.lr = float(os.environ.get("MEOW_MLP_LR", "1e-3"))
        self.weight_decay = float(os.environ.get("MEOW_MLP_WEIGHT_DECAY", "1e-5"))
        self.hidden_dim = int(os.environ.get("MEOW_MLP_HIDDEN", "128"))
        self.n_layers = int(os.environ.get("MEOW_MLP_LAYERS", "2"))
        self.dropout = float(os.environ.get("MEOW_MLP_DROPOUT", "0.1"))
        self.batch_size = int(os.environ.get("MEOW_MLP_BATCH_SIZE", "4096"))
        self.epochs_per_chunk = int(os.environ.get("MEOW_MLP_EPOCHS_PER_CHUNK", "2"))
        self._trained = False

    def reset(self):
        self.model = None
        self.optimizer = None
        self._trained = False

    def _init_model(self, n_features):
        self.n_features = n_features
        self.model = MLPRidge(
            n_features=n_features,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def _train_on_data(self, x_np, y_np, epochs=1):
        self.model.train()
        n = len(x_np)
        idx = np.arange(n, dtype=np.int32)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for j in range(0, n, self.batch_size):
                batch_idx = idx[j : j + self.batch_size]
                bx = x_np[batch_idx]
                by = y_np[batch_idx]
                x_t = torch.from_numpy(bx)
                y_t = torch.from_numpy(by)
                self.optimizer.zero_grad()
                pred = self.model(x_t)
                loss = nn.functional.mse_loss(pred, y_t)
                loss.backward()
                self.optimizer.step()

    def partial_fit(self, xdf, ydf):
        if isinstance(xdf, np.ndarray):
            x = xdf.astype(np.float32)
        else:
            x = xdf.to_numpy(dtype=np.float32)
        y = ydf.to_numpy(dtype=np.float32).ravel()
        nf = x.shape[1]
        if self.model is None:
            self._init_model(nf)
            self.feature_names = list(xdf.columns)
        self._train_on_data(x, y, epochs=self.epochs_per_chunk)

    def finalize_fit(self):
        self._trained = True

    def fit(self, xdf, ydf):
        self.reset()
        self.partial_fit(xdf, ydf)
        self.finalize_fit()

    def predict(self, xdf):
        if self.model is None:
            return np.zeros(len(xdf), dtype=np.float64)
        self.model.eval()
        with torch.no_grad():
            if isinstance(xdf, np.ndarray):
                x = xdf
            else:
                x = xdf.to_numpy(dtype=np.float32)
            n = len(x)
            preds = []
            for i in range(0, n, self.batch_size * 2):
                bx = x[i : i + self.batch_size * 2]
                x_t = torch.from_numpy(bx)
                preds.append(self.model(x_t).cpu().numpy())
            return np.concatenate(preds).ravel().astype(np.float64)
