"""FT-Transformer tabular model: feature tokens + self-attention + MLP head.
Fundamentally different from LGB (greedy tree splits) and Ridge (linear).
Self-attention learns global feature interactions jointly.
"""
from __future__ import annotations

import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class FeatureTokenizer(nn.Module):
    """Embed each scalar feature into a d_model vector using a small MLP."""
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(1, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)
        return self.embed(x.unsqueeze(-1))  # (B, F, d_model)


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer for tabular data."""
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.15,
        mlp_hidden: int = 128,
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
            nn.Linear(d_model, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.LayerNorm(mlp_hidden // 2),
            nn.GELU(),
            nn.Linear(mlp_hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)
        b = x.shape[0]
        tokens = self.tokenizer(x)  # (B, F, d_model)
        cls = self.cls_token.expand(b, -1, -1)  # (B, 1, d_model)
        seq = torch.cat([cls, tokens], dim=1)  # (B, 1+F, d_model)
        h = self.encoder(seq)  # (B, 1+F, d_model)
        cls_out = self.norm(h[:, 0, :])  # (B, d_model)
        return self.head(cls_out).squeeze(-1)


# ---------------------------------------------------------------------------
# Model wrapper with chunked-fit interface
# ---------------------------------------------------------------------------

class TorchModel:
    """Torch FT-Transformer with reservoir-sampled chunked training.

    Interface matches LGBModel/MeowModel: reset / partial_fit / finalize_fit / predict.
    """

    def __init__(self):
        self.max_rows = int(os.environ.get("MEOW_TORCH_MAX_ROWS", "800000"))
        self.d_model = int(os.environ.get("MEOW_TORCH_D_MODEL", "64"))
        self.nhead = int(os.environ.get("MEOW_TORCH_NHEAD", "4"))
        self.num_layers = int(os.environ.get("MEOW_TORCH_NUM_LAYERS", "3"))
        self.dropout = float(os.environ.get("MEOW_TORCH_DROPOUT", "0.15"))
        self.batch_size = int(os.environ.get("MEOW_TORCH_BATCH_SIZE", "4096"))
        self.epochs = int(os.environ.get("MEOW_TORCH_EPOCHS", "30"))
        self.lr = float(os.environ.get("MEOW_TORCH_LR", "1e-3"))
        self.weight_decay = float(os.environ.get("MEOW_TORCH_WEIGHT_DECAY", "1e-4"))
        self.exclude_families = {
            f.strip()
            for f in os.environ.get("MEOW_EXCLUDE_FAMILIES", "cs").split(",")
            if f.strip()
        }
        self._X_reservoir = None
        self._y_reservoir = None
        self._n_accumulated = 0
        self._model = None
        self._feature_names = None
        self._input_mean = None
        self._input_std = None
        self._rng = np.random.RandomState(42)

    # --- Public API ---

    def reset(self):
        self._X_reservoir = None
        self._y_reservoir = None
        self._n_accumulated = 0
        self._model = None
        self._feature_names = None
        self._input_mean = None
        self._input_std = None

    def partial_fit(self, xdf, ydf):
        if self._feature_names is None:
            cols = [c for c in xdf.columns if self._family_of(c) not in self.exclude_families]
            self._feature_names = cols
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
        if self._X_reservoir is None or len(self._y_reservoir) < 2000:
            return
        X = np.asarray(self._X_reservoir, dtype=np.float32)
        y = np.asarray(self._y_reservoir, dtype=np.float32)

        # Standardize
        self._input_mean = X.mean(axis=0, keepdims=True)
        self._input_std = np.std(X, axis=0, keepdims=True).clip(min=1e-8)
        X_norm = (X - self._input_mean) / self._input_std

        # Train/val split
        n = len(y)
        n_val = min(n // 5, 100000)
        idx = self._rng.permutation(n)
        tr_idx = idx[n_val:]
        val_idx = idx[:n_val]

        X_tr = torch.from_numpy(X_norm[tr_idx]).float()
        y_tr = torch.from_numpy(y[tr_idx]).float()
        X_val = torch.from_numpy(X_norm[val_idx]).float()
        y_val = torch.from_numpy(y[val_idx]).float()

        n_features = X.shape[1]
        self._model = FTTransformer(
            n_features=n_features,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self._model.train()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device)
        X_tr, y_tr = X_tr.to(device), y_tr.to(device)
        X_val, y_val = X_val.to(device), y_val.to(device)

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.01
        )

        best_val_loss = float("inf")
        best_state = None
        patience = max(self.epochs // 4, 5)
        no_improve = 0

        n_tr = len(tr_idx)
        for epoch in range(self.epochs):
            perm = torch.randperm(n_tr, device=device)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, n_tr, self.batch_size):
                batch_idx = perm[i : i + self.batch_size]
                xb = X_tr[batch_idx]
                yb = y_tr[batch_idx]

                optimizer.zero_grad()
                pred = self._model(xb)
                loss = F.mse_loss(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Validation
            self._model.eval()
            with torch.no_grad():
                val_pred = self._model(X_val)
                val_loss = F.mse_loss(val_pred, y_val).item()
            self._model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model = self._model.cpu()
        self._model.eval()

        # Free reservoir
        self._X_reservoir = None
        self._y_reservoir = None

    def predict(self, xdf):
        if self._model is None or self._feature_names is None:
            return np.zeros(len(xdf), dtype=np.float64)
        X = xdf[self._feature_names].to_numpy(dtype=np.float32)
        X_norm = (X - self._input_mean) / self._input_std
        X_t = torch.from_numpy(X_norm).float()
        with torch.no_grad():
            pred = self._model(X_t).numpy()
        return pred.astype(np.float64)

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
