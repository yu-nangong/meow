"""BatchEnsemble-style tabular MLP arm for the ridge blend.

References:
- Wen et al., "BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning"
  (ICLR 2020 / arXiv:2002.06715)
- Gorishniy et al., "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"
  (ICLR 2025 / arXiv:2410.24210)

This implementation keeps the idea lightweight for MEOW: one shared MLP with
rank-1 fast weights per ensemble member, bounded reservoir sampling, CPU-only
minibatch training, and chunked prediction.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchEnsembleLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.r = nn.Parameter(torch.empty(ensemble_size, in_features))
        self.s = nn.Parameter(torch.empty(ensemble_size, out_features))
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.normal_(self.r, mean=1.0, std=0.02)
        nn.init.normal_(self.s, mean=1.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D] or [B, K, D] -> [B, K, H]
        if x.dim() == 2:
            hidden = self.linear(x.unsqueeze(1) * self.r.unsqueeze(0))
        else:
            hidden = self.linear(x * self.r.unsqueeze(0))
        return hidden * self.s.unsqueeze(0)


class BatchEnsembleMLP(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, ensemble_size: int, dropout: float):
        super().__init__()
        self.layer1 = BatchEnsembleLinear(n_features, hidden_dim, ensemble_size)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.layer2 = BatchEnsembleLinear(hidden_dim, hidden_dim, ensemble_size)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.layer1(x)
        hidden = self.norm1(hidden)
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.layer2(hidden)
        hidden = self.norm2(hidden)
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)
        return self.head(hidden).squeeze(-1)


class TabMModel:
    def __init__(self):
        self.max_rows = int(os.environ.get("MEOW_TABM_MAX_ROWS", "120000"))
        self.hidden_dim = int(os.environ.get("MEOW_TABM_HIDDEN_DIM", "64"))
        self.ensemble_size = int(os.environ.get("MEOW_TABM_ENSEMBLE_SIZE", "4"))
        self.dropout = float(os.environ.get("MEOW_TABM_DROPOUT", "0.10"))
        self.batch_size = int(os.environ.get("MEOW_TABM_BATCH_SIZE", "2048"))
        self.epochs = int(os.environ.get("MEOW_TABM_EPOCHS", "8"))
        self.lr = float(os.environ.get("MEOW_TABM_LR", "1e-3"))
        self.weight_decay = float(os.environ.get("MEOW_TABM_WEIGHT_DECAY", "1e-4"))
        self.patience = int(os.environ.get("MEOW_TABM_PATIENCE", "2"))
        self.predict_batch_size = int(os.environ.get("MEOW_TABM_PREDICT_BATCH_SIZE", "8192"))
        self.torch_threads = int(os.environ.get("MEOW_TABM_THREADS", "2"))
        self.exclude_families = {
            f.strip()
            for f in os.environ.get(
                "MEOW_TABM_EXCLUDE_FAMILIES",
                "cs,time_interaction,u_interaction,time_sq_interaction,u_sq_interaction",
            ).split(",")
            if f.strip()
        }
        self.exclude_patterns = tuple(
            pattern.strip()
            for pattern in os.environ.get(
                "MEOW_TABM_EXCLUDE_PATTERNS",
                "midpx_level_rank_cs,lastpx_level_rank_cs,high_level_rank_cs,"
                "low_level_rank_cs,open_level_rank_cs,bid0_level_rank_cs,ask0_level_rank_cs",
            ).split(",")
            if pattern.strip()
        )
        self._X_reservoir = None
        self._y_reservoir = None
        self._n_accumulated = 0
        self._feature_names = None
        self._mean = None
        self._std = None
        self._model = None
        self._rng = np.random.RandomState(42)

    def reset(self):
        self._X_reservoir = None
        self._y_reservoir = None
        self._n_accumulated = 0
        self._feature_names = None
        self._mean = None
        self._std = None
        self._model = None

    def partial_fit(self, xdf, ydf):
        if self._feature_names is None:
            self._feature_names = [col for col in xdf.columns if self._keep_column(col)]
        x = xdf[self._feature_names].to_numpy(dtype=np.float32)
        y = ydf.to_numpy(dtype=np.float32).ravel()
        n = len(x)
        if self._X_reservoir is None:
            take = min(n, self.max_rows)
            self._X_reservoir = x[:take].copy()
            self._y_reservoir = y[:take].copy()
            self._n_accumulated = n
            return
        capacity = self._X_reservoir.shape[0]
        if capacity < self.max_rows:
            take = min(n, self.max_rows - capacity)
            if take > 0:
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
        if self._X_reservoir is None or len(self._y_reservoir) < 4000:
            return
        torch.set_num_threads(self.torch_threads)
        x = np.asarray(self._X_reservoir, dtype=np.float32)
        y = np.asarray(self._y_reservoir, dtype=np.float32)
        self._mean = x.mean(axis=0)
        self._std = np.maximum(x.std(axis=0), 1e-6)
        x = (x - self._mean) / self._std

        idx = self._rng.permutation(len(y))
        n_val = min(max(len(y) // 10, 2048), 20000)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        if len(train_idx) == 0:
            train_idx = idx
            val_idx = idx[: min(len(idx), 2048)]

        x_train = torch.from_numpy(x[train_idx])
        y_train = torch.from_numpy(y[train_idx])
        x_val = torch.from_numpy(x[val_idx])
        y_val = torch.from_numpy(y[val_idx])

        self._model = BatchEnsembleMLP(
            n_features=x.shape[1],
            hidden_dim=self.hidden_dim,
            ensemble_size=self.ensemble_size,
            dropout=self.dropout,
        )
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss = float("inf")
        best_state = None
        bad_epochs = 0
        n_train = len(train_idx)
        for _ in range(self.epochs):
            self._model.train()
            perm = torch.randperm(n_train)
            for start in range(0, n_train, self.batch_size):
                batch = perm[start : start + self.batch_size]
                xb = x_train[batch]
                yb = y_train[batch].unsqueeze(1)
                pred_members = self._model(xb)
                loss = F.mse_loss(pred_members, yb.expand(-1, self.ensemble_size))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
            self._model.eval()
            with torch.no_grad():
                val_pred = self._model(x_val).mean(dim=1)
                val_loss = F.mse_loss(val_pred, y_val).item()
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
        self._model.eval()
        self._X_reservoir = None
        self._y_reservoir = None

    def predict(self, xdf):
        if self._model is None or self._feature_names is None:
            return np.zeros(len(xdf), dtype=np.float64)
        x = xdf[self._feature_names].to_numpy(dtype=np.float32)
        x = (x - self._mean) / self._std
        preds = []
        with torch.no_grad():
            for start in range(0, len(x), self.predict_batch_size):
                xb = torch.from_numpy(x[start : start + self.predict_batch_size])
                pred = self._model(xb).mean(dim=1).numpy()
                preds.append(pred)
        return np.concatenate(preds).astype(np.float64, copy=False)

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
