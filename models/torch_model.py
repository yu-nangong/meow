"""Compact FT-style tabular transformer for low-weight ensemble diversification."""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(1, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x.unsqueeze(-1))


class FTTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(batch, -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(self.norm(encoded[:, 0, :])).squeeze(-1)


class TorchModel:
    """Reservoir-trained FT-style arm sized to be a weak ensemble member."""

    def __init__(self):
        self.max_rows = int(os.environ.get("MEOW_TORCH_MAX_ROWS", "120000"))
        self.d_model = int(os.environ.get("MEOW_TORCH_D_MODEL", "32"))
        self.nhead = int(os.environ.get("MEOW_TORCH_NHEAD", "4"))
        self.num_layers = int(os.environ.get("MEOW_TORCH_NUM_LAYERS", "2"))
        self.dropout = float(os.environ.get("MEOW_TORCH_DROPOUT", "0.10"))
        self.batch_size = int(os.environ.get("MEOW_TORCH_BATCH_SIZE", "2048"))
        self.epochs = int(os.environ.get("MEOW_TORCH_EPOCHS", "8"))
        self.lr = float(os.environ.get("MEOW_TORCH_LR", "8e-4"))
        self.weight_decay = float(os.environ.get("MEOW_TORCH_WEIGHT_DECAY", "1e-4"))
        self.pred_batch_size = int(os.environ.get("MEOW_TORCH_PRED_BATCH_SIZE", "8192"))
        self.exclude_families = {
            item.strip()
            for item in os.environ.get("MEOW_TORCH_EXCLUDE_FAMILIES", "cs").split(",")
            if item.strip()
        }
        self.exclude_patterns = tuple(
            pattern.strip()
            for pattern in os.environ.get("MEOW_TORCH_EXCLUDE_PATTERNS", "").split(",")
            if pattern.strip()
        )
        self._rng = np.random.RandomState(42)
        self.reset()

    def reset(self):
        self._X_reservoir = None
        self._y_reservoir = None
        self._n_accumulated = 0
        self._feature_names = None
        self._input_mean = None
        self._input_std = None
        self._target_mean = 0.0
        self._target_std = 1.0
        self._model = None

    def partial_fit(self, xdf, ydf):
        if self._feature_names is None:
            self._feature_names = [col for col in xdf.columns if self._keep_column(col)]
        if not self._feature_names:
            return
        x = xdf.loc[:, self._feature_names].to_numpy(dtype=np.float32, copy=False)
        y = ydf.to_numpy(dtype=np.float32).ravel()
        n_rows = len(y)
        if self._X_reservoir is None:
            take = min(n_rows, self.max_rows)
            self._X_reservoir = x[:take].copy()
            self._y_reservoir = y[:take].copy()
            self._n_accumulated = n_rows
            return
        capacity = self._X_reservoir.shape[0]
        if capacity < self.max_rows:
            take = min(n_rows, self.max_rows - capacity)
            self._X_reservoir = np.concatenate([self._X_reservoir, x[:take]], axis=0)
            self._y_reservoir = np.concatenate([self._y_reservoir, y[:take]], axis=0)
        else:
            for idx in range(n_rows):
                draw = self._rng.randint(0, self._n_accumulated + idx + 1)
                if draw < capacity:
                    self._X_reservoir[draw] = x[idx]
                    self._y_reservoir[draw] = y[idx]
        self._n_accumulated += n_rows

    def finalize_fit(self):
        if self._X_reservoir is None or len(self._y_reservoir) < 4096:
            return
        x = self._X_reservoir.astype(np.float32, copy=False)
        y = self._y_reservoir.astype(np.float32, copy=False)
        self._input_mean = x.mean(axis=0, keepdims=True).astype(np.float32, copy=False)
        self._input_std = np.std(x, axis=0, keepdims=True).astype(np.float32, copy=False)
        self._input_std = np.maximum(self._input_std, 1e-6)
        x = (x - self._input_mean) / self._input_std
        self._target_mean = float(y.mean())
        self._target_std = float(max(y.std(), 1e-6))
        y = (y - self._target_mean) / self._target_std

        n_rows = len(y)
        val_size = min(max(n_rows // 10, 2048), 16384)
        perm = self._rng.permutation(n_rows)
        val_idx = perm[:val_size]
        tr_idx = perm[val_size:]

        model = FTTransformer(
            n_features=x.shape[1],
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_state = None
        best_val = float("inf")
        patience = 3
        stale = 0

        for _ in range(self.epochs):
            model.train()
            epoch_perm = self._rng.permutation(len(tr_idx))
            for start in range(0, len(epoch_perm), self.batch_size):
                batch_ids = tr_idx[epoch_perm[start : start + self.batch_size]]
                xb = torch.from_numpy(x[batch_ids]).float()
                yb = torch.from_numpy(y[batch_ids]).float()
                optimizer.zero_grad()
                pred = model(xb)
                loss = F.mse_loss(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            val_losses = []
            with torch.no_grad():
                for start in range(0, len(val_idx), self.pred_batch_size):
                    batch_ids = val_idx[start : start + self.pred_batch_size]
                    xb = torch.from_numpy(x[batch_ids]).float()
                    yb = torch.from_numpy(y[batch_ids]).float()
                    val_losses.append(float(F.mse_loss(model(xb), yb).item()))
            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            if val_loss < best_val:
                best_val = val_loss
                best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self._model = model
        self._X_reservoir = None
        self._y_reservoir = None

    def predict(self, xdf):
        if self._model is None or not self._feature_names:
            return np.zeros(len(xdf), dtype=np.float64)
        x = xdf.loc[:, self._feature_names].to_numpy(dtype=np.float32, copy=False)
        x = (x - self._input_mean) / self._input_std
        preds = np.empty(len(x), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(x), self.pred_batch_size):
                end = start + self.pred_batch_size
                xb = torch.from_numpy(x[start:end]).float()
                preds[start:end] = self._model(xb).numpy()
        preds = preds * self._target_std + self._target_mean
        return preds.astype(np.float64)

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
