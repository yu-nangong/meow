"""Neural network residual corrector: small MLP trained on post-pipeline residuals.

Fundamentally different from all existing models:
- Operates on residuals, not raw target — easier learning problem.
- Uses conservative blend weight — cannot destroy the base signal.
- Global z-score normalization on reservoir, not per-chunk.
- Proper batch size (1024) and epoch count (15) vs failed Pearson NN.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn


class ResidualNet(nn.Module):
    """Skip-connected MLP for residual correction."""
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(n_features)
        self.fc1 = nn.Linear(n_features, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.bn2 = nn.BatchNorm1d(hidden // 2)
        self.fc3 = nn.Linear(hidden // 2, hidden // 4)
        self.bn3 = nn.BatchNorm1d(hidden // 4)
        self.head = nn.Linear(hidden // 4, 1)
        self.dropout = nn.Dropout(0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)
        h = torch.relu(self.bn1(self.fc1(x)))
        h = self.dropout(h)
        h = torch.relu(self.bn2(self.fc2(h)))
        h = self.dropout(h)
        h = torch.relu(self.bn3(self.fc3(h)))
        return self.head(h).squeeze(-1)


class NnResidualModel:
    """Scikit-learn-compatible interface for neural residual correction."""

    def __init__(self):
        self.lr = float(os.environ.get("MEOW_NNR_LR", "3e-4"))
        self.epochs = int(os.environ.get("MEOW_NNR_EPOCHS", "15"))
        self.batch_size = int(os.environ.get("MEOW_NNR_BATCH_SIZE", "1024"))
        self.hidden = int(os.environ.get("MEOW_NNR_HIDDEN", "64"))
        self.weight_decay = float(os.environ.get("MEOW_NNR_WD", "1e-6"))
        self.max_rows = int(os.environ.get("MEOW_NNR_MAX_ROWS", "200000"))
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._n_features = None
        self._x_res = None
        self._y_res = None
        self._n_seen = 0
        self._rng = np.random.RandomState(42)
        self._x_mean = None
        self._x_scale = None
        self._y_std = 1.0

    def reset(self):
        self._model = None
        self._n_features = None
        self._x_res = None
        self._y_res = None
        self._n_seen = 0
        self._x_mean = None
        self._x_scale = None
        self._y_std = 1.0

    def partial_fit(self, xdf, resid):
        """Accumulate features and residuals via reservoir sampling."""
        x = xdf.to_numpy(dtype=np.float32)
        r = np.asarray(resid, dtype=np.float32)
        n = len(x)
        if n == 0:
            return

        if self._x_res is None:
            take = min(n, self.max_rows)
            self._x_res = x[:take].copy()
            self._y_res = r[:take].copy()
            self._n_seen = n
            self._n_features = x.shape[1]
            return

        capacity = self._x_res.shape[0]
        if capacity < self.max_rows:
            take = min(n, self.max_rows - capacity)
            self._x_res = np.concatenate([self._x_res, x[:take]], axis=0)
            self._y_res = np.concatenate([self._y_res, r[:take]], axis=0)
            start = take
        else:
            start = 0
        for i in range(start, n):
            j = self._rng.randint(0, self._n_seen + i + 1)
            if j < self.max_rows:
                self._x_res[j] = x[i]
                self._y_res[j] = r[i]
        self._n_seen += n

    def finalize_fit(self):
        if self._y_res is None or len(self._y_res) < 5000:
            return
        # Global normalization stats
        self._x_mean = self._x_res.mean(axis=0, dtype=np.float64)
        self._x_scale = self._x_res.std(axis=0, dtype=np.float64)
        self._x_scale = np.where(self._x_scale > 1e-6, self._x_scale, 1.0)
        self._y_std = float(np.std(self._y_res))
        if self._y_std < 1e-8:
            self._y_std = 1.0

        x_norm = ((self._x_res - self._x_mean) / self._x_scale).astype(np.float32, copy=False)
        y_norm = (self._y_res / self._y_std).astype(np.float32, copy=False)

        self._model = ResidualNet(self._n_features, self.hidden).to(self._device)
        opt = torch.optim.AdamW(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        n_samples = len(y_norm)

        self._model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n_samples, device="cpu")
            epoch_loss = 0.0
            for i in range(0, n_samples, self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb = torch.from_numpy(x_norm[idx]).float().to(self._device)
                yb = torch.from_numpy(y_norm[idx]).float().to(self._device)
                opt.zero_grad()
                pred = self._model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * len(idx)
            scheduler.step()
        # Free reservoir
        self._x_res = None
        self._y_res = None

    def predict(self, xdf):
        """Predict residual correction. Returns zeros if model not trained."""
        if self._model is None or self._x_mean is None:
            return np.zeros(len(xdf), dtype=np.float64)
        x = xdf.to_numpy(dtype=np.float32)
        x_norm = ((x - self._x_mean) / self._x_scale).astype(np.float32, copy=False)

        self._model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(x_norm), self.batch_size):
                xb = torch.from_numpy(x_norm[i:i+self.batch_size]).float().to(self._device)
                p = self._model(xb).cpu().numpy()
                preds.append(p)
        raw = np.concatenate(preds).astype(np.float64)
        return raw * self._y_std
