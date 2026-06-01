"""Lightweight temporal model over short raw LOB history windows."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


class _TinyTemporalCNN(nn.Module):
    def __init__(self, n_features: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.transpose(1, 2)).squeeze(-1)
        return self.head(h).squeeze(-1)


class LagSequenceModel:
    def __init__(self):
        raw_cols = os.environ.get(
            "MEOW_SEQ_RAW_COLS",
            "bid0,ask0,bsize0,asize0,tradeBuyQty,tradeSellQty,tradeBuyTurnover,"
            "tradeSellTurnover,midpx,lastpx",
        )
        self.raw_cols = [col.strip() for col in raw_cols.split(",") if col.strip()]
        self.lookback = int(os.environ.get("MEOW_SEQ_LOOKBACK", "8"))
        self.max_rows = int(os.environ.get("MEOW_SEQ_MAX_ROWS", "150000"))
        self.hidden = int(os.environ.get("MEOW_SEQ_HIDDEN", "24"))
        self.epochs = int(os.environ.get("MEOW_SEQ_EPOCHS", "3"))
        self.batch_size = int(os.environ.get("MEOW_SEQ_BATCH_SIZE", "2048"))
        self.learning_rate = float(os.environ.get("MEOW_SEQ_LR", "8e-4"))
        self.weight_decay = float(os.environ.get("MEOW_SEQ_WEIGHT_DECAY", "1e-5"))
        self.val_frac = float(os.environ.get("MEOW_SEQ_VAL_FRAC", "0.15"))
        self.device = os.environ.get("MEOW_SEQ_DEVICE", "cpu")
        self._rng = np.random.RandomState(42)
        self.reset()

    def reset(self):
        self._X_reservoir = None
        self._y_reservoir = None
        self._n_seen = 0
        self._model = None
        self._mean = None
        self._std = None

    def partial_fit(self, raw: pd.DataFrame):
        seq_x, seq_y, _ = self._build_sequences(raw)
        if len(seq_y) == 0:
            return
        self._update_reservoir(seq_x, seq_y)

    def finalize_fit(self):
        if self._X_reservoir is None or len(self._y_reservoir) < 5000:
            return
        x = np.nan_to_num(self._X_reservoir.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(self._y_reservoir.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

        self._mean = x.mean(axis=(0, 1), keepdims=True)
        self._std = np.clip(x.std(axis=(0, 1), keepdims=True), 1e-6, None)
        x = (x - self._mean) / self._std

        n = len(y)
        n_val = max(int(n * self.val_frac), 2048)
        n_val = min(n_val, n // 4)
        perm = self._rng.permutation(n)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        if len(tr_idx) < 2048 or len(val_idx) < 1024:
            return

        x_tr = torch.from_numpy(x[tr_idx])
        y_tr = torch.from_numpy(y[tr_idx])
        x_val = torch.from_numpy(x[val_idx])
        y_val = torch.from_numpy(y[val_idx])

        model = _TinyTemporalCNN(n_features=x.shape[2], hidden=self.hidden).to(self.device)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        best_state = None
        best_val = float("inf")
        stale = 0

        for _ in range(self.epochs):
            model.train()
            order = torch.randperm(len(tr_idx))
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                xb = x_tr[batch].to(self.device)
                yb = y_tr[batch].to(self.device)
                opt.zero_grad()
                pred = model(xb)
                loss = F.mse_loss(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            with torch.no_grad():
                val_pred = []
                for start in range(0, len(val_idx), self.batch_size):
                    xb = x_val[start : start + self.batch_size].to(self.device)
                    val_pred.append(model(xb).cpu())
                pred = torch.cat(val_pred)
                val_loss = F.mse_loss(pred, y_val).item()
            if val_loss + 1e-12 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= 2:
                    break

        if best_state is None:
            return
        model.load_state_dict(best_state)
        self._model = model.cpu().eval()
        self._X_reservoir = None
        self._y_reservoir = None

    def predict(self, raw: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            return np.zeros(len(raw), dtype=np.float64)
        seq_x, _, row_ids = self._build_sequences(raw, include_target=False)
        if len(seq_x) == 0:
            return np.zeros(len(raw), dtype=np.float64)
        seq_x = np.nan_to_num(seq_x, nan=0.0, posinf=0.0, neginf=0.0)
        seq_x = ((seq_x - self._mean) / self._std).astype(np.float32, copy=False)
        out = np.zeros(len(raw), dtype=np.float64)
        preds = []
        with torch.no_grad():
            for start in range(0, len(seq_x), self.batch_size):
                xb = torch.from_numpy(seq_x[start : start + self.batch_size])
                preds.append(self._model(xb).numpy())
        out[row_ids] = np.concatenate(preds).astype(np.float64, copy=False)
        return out

    def _build_sequences(self, raw: pd.DataFrame, include_target: bool = True):
        cols = [c for c in self.raw_cols if c in raw.columns]
        empty_x = np.zeros((0, self.lookback, len(self.raw_cols)), dtype=np.float32)
        empty_y = np.zeros(0, dtype=np.float32)
        empty_ids = np.zeros(0, dtype=np.int64)
        if len(cols) != len(self.raw_cols):
            return empty_x, empty_y, empty_ids
        raw = raw.reset_index(drop=True).copy()
        raw["_row_id"] = np.arange(len(raw), dtype=np.int64)
        raw = raw.sort_values(["date", "symbol", "interval"], kind="mergesort")
        x_parts = []
        y_parts = []
        id_parts = []
        for _, grp in raw.groupby(["date", "symbol"], sort=False):
            arr = grp.loc[:, cols].to_numpy(dtype=np.float32, copy=False)
            if len(arr) < self.lookback:
                continue
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            windows = np.lib.stride_tricks.sliding_window_view(arr, window_shape=self.lookback, axis=0)
            windows = np.swapaxes(windows, 1, 2)
            x_parts.append(np.ascontiguousarray(windows))
            id_parts.append(grp["_row_id"].to_numpy(dtype=np.int64, copy=False)[self.lookback - 1 :])
            if include_target:
                y_parts.append(grp["fret12"].to_numpy(dtype=np.float32, copy=False)[self.lookback - 1 :])
        if not x_parts:
            return empty_x, empty_y, empty_ids
        x = np.concatenate(x_parts, axis=0)
        if not include_target:
            return x, empty_y, np.concatenate(id_parts, axis=0)
        return x, np.concatenate(y_parts, axis=0), np.concatenate(id_parts, axis=0)

    def _update_reservoir(self, x: np.ndarray, y: np.ndarray):
        n = len(y)
        if self._X_reservoir is None:
            take = min(n, self.max_rows)
            self._X_reservoir = x[:take].copy()
            self._y_reservoir = y[:take].copy()
            self._n_seen = n
            return
        capacity = len(self._y_reservoir)
        start = 0
        if capacity < self.max_rows:
            take = min(n, self.max_rows - capacity)
            self._X_reservoir = np.concatenate([self._X_reservoir, x[:take]], axis=0)
            self._y_reservoir = np.concatenate([self._y_reservoir, y[:take]], axis=0)
            start = take
        for i in range(start, n):
            j = self._rng.randint(0, self._n_seen + i + 1)
            if j < self.max_rows:
                self._X_reservoir[j] = x[i]
                self._y_reservoir[j] = y[i]
        self._n_seen += n
