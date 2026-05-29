from __future__ import annotations

import math
import os

import numpy as np
import torch
from torch import nn


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ResidualMLPTrainer:
    def __init__(self, input_dim: int):
        hidden_dim = int(os.environ.get("MEOW_RESIDUAL_HIDDEN_DIM", "96"))
        hidden_dim = max(hidden_dim, 8)
        dropout = float(os.environ.get("MEOW_RESIDUAL_DROPOUT", "0.05"))
        self.batch_size = int(os.environ.get("MEOW_RESIDUAL_BATCH_SIZE", "16384"))
        self.lr = float(os.environ.get("MEOW_RESIDUAL_LR", "0.001"))
        self.weight_decay = float(os.environ.get("MEOW_RESIDUAL_WEIGHT_DECAY", "0.0001"))
        self.clip_grad_norm = float(os.environ.get("MEOW_RESIDUAL_CLIP_GRAD_NORM", "1.0"))
        self.device = torch.device(os.environ.get("MEOW_RESIDUAL_DEVICE", "cpu"))
        num_threads = int(os.environ.get("MEOW_TORCH_NUM_THREADS", "1"))
        torch.set_num_threads(max(num_threads, 1))
        self.model = ResidualMLP(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.loss_fn = nn.MSELoss()

    def fit_chunk(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(x) == 0:
            return
        self.model.train()
        n_rows = x.shape[0]
        step = self.batch_size
        for start in range(0, n_rows, step):
            stop = min(start + step, n_rows)
            xb = torch.from_numpy(np.ascontiguousarray(x[start:stop])).to(self.device)
            yb = torch.from_numpy(np.ascontiguousarray(y[start:stop])).to(self.device)
            pred = self.model(xb)
            loss = self.loss_fn(pred, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return np.zeros(0, dtype=np.float32)
        self.model.eval()
        out = np.empty(x.shape[0], dtype=np.float32)
        step = self.batch_size
        for start in range(0, x.shape[0], step):
            stop = min(start + step, x.shape[0])
            xb = torch.from_numpy(np.ascontiguousarray(x[start:stop])).to(self.device)
            out[start:stop] = self.model(xb).cpu().numpy().astype(np.float32, copy=False)
        return out
