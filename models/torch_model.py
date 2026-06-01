"""Compact FT-style tabular transformer arm.

Reference: Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data"
(arXiv:2106.11959). This keeps the idea lightweight enough for the MEOW
benchmark by using a small feature tokenizer, a shallow transformer encoder,
and bounded reservoir sampling.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NumericalFeatureTokenizer(nn.Module):
    """Map each scalar feature to a token via learned scale and bias."""

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformerBackbone(nn.Module):
    def __init__(self, n_features: int, d_token: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.tokenizer = NumericalFeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        hidden = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.out_norm(hidden[:, 0])


class FTTransformerRegressor(nn.Module):
    def __init__(self, n_features: int, d_token: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.backbone = FTTransformerBackbone(n_features, d_token, n_heads, n_layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.LayerNorm(d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(-1)


class TorchModel:
    """Chunked tabular transformer with bounded-memory training."""

    def __init__(self):
        self.max_rows = int(os.environ.get("MEOW_TORCH_MAX_ROWS", "300000"))
        self.d_token = int(os.environ.get("MEOW_TORCH_D_TOKEN", "32"))
        self.n_heads = int(os.environ.get("MEOW_TORCH_N_HEADS", "4"))
        self.n_layers = int(os.environ.get("MEOW_TORCH_N_LAYERS", "2"))
        self.dropout = float(os.environ.get("MEOW_TORCH_DROPOUT", "0.10"))
        self.batch_size = int(os.environ.get("MEOW_TORCH_BATCH_SIZE", "8192"))
        self.epochs = int(os.environ.get("MEOW_TORCH_EPOCHS", "10"))
        self.lr = float(os.environ.get("MEOW_TORCH_LR", "1e-3"))
        self.weight_decay = float(os.environ.get("MEOW_TORCH_WEIGHT_DECAY", "1e-4"))
        self.patience = int(os.environ.get("MEOW_TORCH_PATIENCE", "3"))
        self.torch_threads = int(os.environ.get("MEOW_TORCH_THREADS", "2"))
        self.exclude_families = {
            f.strip()
            for f in os.environ.get(
                "MEOW_TORCH_EXCLUDE_FAMILIES",
                "cs,time_interaction,u_interaction,time_sq_interaction,u_sq_interaction",
            ).split(",")
            if f.strip()
        }
        self.exclude_patterns = tuple(
            pattern.strip()
            for pattern in os.environ.get(
                "MEOW_TORCH_EXCLUDE_PATTERNS",
                "midpx_level_rank_cs,lastpx_level_rank_cs,high_level_rank_cs,"
                "low_level_rank_cs,open_level_rank_cs,bid0_level_rank_cs,"
                "ask0_level_rank_cs",
            ).split(",")
            if pattern.strip()
        )
        self._X_reservoir = None
        self._y_reservoir = None
        self._n_accumulated = 0
        self._model = None
        self._feature_names = None
        self._mean = None
        self._std = None
        self._rng = np.random.RandomState(42)

    def reset(self):
        self._X_reservoir = None
        self._y_reservoir = None
        self._n_accumulated = 0
        self._model = None
        self._feature_names = None
        self._mean = None
        self._std = None

    def partial_fit(self, xdf, ydf):
        if self._feature_names is None:
            self._feature_names = [col for col in xdf.columns if self._keep_column(col)]
        x = xdf[self._feature_names].to_numpy(dtype=np.float32)
        y = ydf.to_numpy(dtype=np.float32).ravel()
        n = len(x)
        if self._X_reservoir is None:
            self._X_reservoir = x.copy()
            self._y_reservoir = y.copy()
            self._n_accumulated = n
            return
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
        if self._X_reservoir is None or len(self._y_reservoir) < 2000:
            return
        torch.set_num_threads(self.torch_threads)
        x = np.asarray(self._X_reservoir, dtype=np.float32)
        y = np.asarray(self._y_reservoir, dtype=np.float32)
        self._mean = x.mean(axis=0, keepdims=True)
        self._std = np.maximum(x.std(axis=0, keepdims=True), 1e-6)
        x = (x - self._mean) / self._std
        idx = self._rng.permutation(len(y))
        n_val = min(max(len(y) // 10, 4096), 50000)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        x_train = torch.from_numpy(x[train_idx])
        y_train = torch.from_numpy(y[train_idx])
        x_val = torch.from_numpy(x[val_idx])
        y_val = torch.from_numpy(y[val_idx])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = FTTransformerRegressor(
            n_features=x.shape[1],
            d_token=self.d_token,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(device)
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss = float("inf")
        best_state = None
        bad_epochs = 0
        n_train = len(train_idx)
        for _ in range(self.epochs):
            self._model.train()
            perm = torch.randperm(n_train, device=device)
            for start in range(0, n_train, self.batch_size):
                batch = perm[start : start + self.batch_size]
                pred = self._model(x_train[batch])
                loss = F.mse_loss(pred, y_train[batch])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
            self._model.eval()
            with torch.no_grad():
                val_loss = F.mse_loss(self._model(x_val), y_val).item()
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model = self._model.cpu()
        self._model.eval()
        self._X_reservoir = None
        self._y_reservoir = None

    def predict(self, xdf):
        if self._model is None or self._feature_names is None:
            return np.zeros(len(xdf), dtype=np.float64)
        x = xdf[self._feature_names].to_numpy(dtype=np.float32)
        x = (x - self._mean) / self._std
        with torch.no_grad():
            pred = self._model(torch.from_numpy(x)).numpy()
        return pred.astype(np.float64)

    def _keep_column(self, name):
        if self.exclude_patterns and any(pattern in name for pattern in self.exclude_patterns):
            return False
        return self._family_of(name) not in self.exclude_families

    @staticmethod
    def _family_of(name):
        if name.endswith("_x_time_sq"):
            return "time_sq_interaction"
        if name.endswith("_x_u_sq"):
            return "u_sq_interaction"
        if name.endswith("_x_time"):
            return "time_interaction"
        if name.endswith("_x_u"):
            return "u_interaction"
        if name.endswith("_rank_cs"):
            return "rank"
        if name.endswith("_cs"):
            return "cs"
        if name.startswith("interval_"):
            return "time_basis"
        return "raw"
