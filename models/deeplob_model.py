"""
DeepLOB model for LOB microstructure prediction.

Processes LOB data as structured tensors:
  - (T, C, L) where T=lookback, C=channels, L=price levels
  - Conv1d along level axis for spatial LOB structure
  - LSTM along time axis for temporal dynamics
  - Linear head for fret12 prediction

Reference: Zhang et al., "DeepLOB: Deep Convolutional Neural Networks for
Limit Order Books", 2019. Adapted for 4-level aggregated LOB data.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# LOB tensor columns (4 levels x 4 channels)
# ---------------------------------------------------------------------------
LOB_BID_PRICE_COLS = ["bid0", "bid4", "bid9", "bid19"]
LOB_ASK_PRICE_COLS = ["ask0", "ask4", "ask9", "ask19"]
LOB_BID_SIZE_COLS = ["bsize0", "bsize0_4", "bsize5_9", "bsize10_19"]
LOB_ASK_SIZE_COLS = ["asize0", "asize0_4", "asize5_9", "asize10_19"]
LOB_CHANNEL_COLS = [LOB_BID_PRICE_COLS, LOB_ASK_PRICE_COLS, LOB_BID_SIZE_COLS, LOB_ASK_SIZE_COLS]

# Global features per timestep (not level-structured)
LOB_GLOBAL_COLS = ["midpx", "lastpx", "tradeBuyQty", "tradeSellQty",
                    "tradeBuyTurnover", "tradeSellTurnover"]

N_LEVELS = 4
N_CHANNELS = 4  # bid_price, ask_price, bid_size, ask_size


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------
class DeepLOBNet(nn.Module):
    """Conv1d on LOB levels -> LSTM over time -> scalar forecast."""

    def __init__(self, n_levels: int = N_LEVELS, n_channels: int = N_CHANNELS,
                 n_global: int = 0,
                 conv_hidden: int = 32, lstm_hidden: int = 64,
                 lstm_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_levels = n_levels
        self.n_channels = n_channels
        self.n_global = n_global

        # Level-axis conv: capture LOB shape across levels
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, conv_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_hidden, conv_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        conv_flat_dim = conv_hidden * n_levels + n_global

        # Temporal modelling
        self.lstm = nn.LSTM(
            input_size=conv_flat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Head
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, 1),
        )

    def forward(self, x_lob, x_global=None):
        """
        x_lob:   (B, T, C, L)  — LOB tensor
        x_global: (B, T, G) or None
        Returns: (B,) — scalar forecast per window
        """
        B, T, C, L = x_lob.shape
        # Merge batch+time for conv
        x_lob = x_lob.reshape(B * T, C, L)          # (B*T, C, L)
        h = self.conv(x_lob)                         # (B*T, conv_hidden, L)
        h = h.reshape(B, T, -1)                      # (B, T, conv_hidden*L)

        if x_global is not None and self.n_global > 0:
            h = torch.cat([h, x_global], dim=-1)

        out, (hn, _) = self.lstm(h)                  # (B, T, lstm_hidden)
        last = out[:, -1, :]                          # (B, lstm_hidden)
        return self.head(last).squeeze(-1)            # (B,)


# ---------------------------------------------------------------------------
# Trainer class (mirrors LagMLPSequenceModel interface)
# ---------------------------------------------------------------------------
class DeepLOBModel:
    """LOB tensor sequence model for post-pipeline signal."""

    def __init__(self):
        self.lookback = int(os.environ.get("MEOW_DLOB_LOOKBACK", "16"))
        self.max_rows = int(os.environ.get("MEOW_DLOB_MAX_ROWS", "150000"))
        self.lstm_hidden = int(os.environ.get("MEOW_DLOB_HIDDEN", "64"))
        self.lstm_layers = int(os.environ.get("MEOW_DLOB_LAYERS", "2"))
        self.conv_hidden = int(os.environ.get("MEOW_DLOB_CONV", "32"))
        self.epochs = int(os.environ.get("MEOW_DLOB_EPOCHS", "15"))
        self.batch_size = int(os.environ.get("MEOW_DLOB_BATCH", "512"))
        self.lr = float(os.environ.get("MEOW_DLOB_LR", "5e-4"))
        self.weight_decay = float(os.environ.get("MEOW_DLOB_WD", "1e-5"))
        self.dropout = float(os.environ.get("MEOW_DLOB_DROP", "0.15"))
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Reservoir
        self._res_x_lob = None
        self._res_x_glob = None
        self._res_y = None
        self._n_seen = 0
        self._rng = np.random.RandomState(42)

        # Normalization stats
        self._lob_mean = None
        self._lob_scale = None
        self._glob_mean = None
        self._glob_scale = None
        self._y_mean = 0.0
        self._y_std = 1.0

        self._model = None
        self._active_level_cols = None
        self._active_global_cols = None
        self._midpx_idx = None

    # -- Interface ----------------------------------------------------------

    def reset(self):
        self._res_x_lob = None
        self._res_x_glob = None
        self._res_y = None
        self._n_seen = 0
        self._lob_mean = None
        self._lob_scale = None
        self._glob_mean = None
        self._glob_scale = None
        self._model = None
        self._active_level_cols = None
        self._active_global_cols = None
        self._midpx_idx = None

    def partial_fit(self, raw):
        lob, glob, targets, _masks = self._build_train_examples(raw)
        if len(targets) == 0:
            return
        n = len(targets)
        if self._res_x_lob is None:
            take = min(n, self.max_rows)
            self._res_x_lob = lob[:take].copy()
            self._res_x_glob = glob[:take].copy()
            self._res_y = targets[:take].copy()
            self._n_seen = n
            return
        capacity = self._res_x_lob.shape[0]
        if capacity < self.max_rows:
            take = min(n, self.max_rows - capacity)
            self._res_x_lob = np.concatenate([self._res_x_lob, lob[:take]], axis=0)
            self._res_x_glob = np.concatenate([self._res_x_glob, glob[:take]], axis=0)
            self._res_y = np.concatenate([self._res_y, targets[:take]], axis=0)
            start = take
        else:
            start = 0
        for i in range(start, n):
            j = self._rng.randint(0, self._n_seen + i + 1)
            if j < self.max_rows:
                self._res_x_lob[j] = lob[i]
                self._res_x_glob[j] = glob[i]
                self._res_y[j] = targets[i]
        self._n_seen += n

    def finalize_fit(self):
        if self._res_y is None or len(self._res_y) < 1000:
            return
        # Fit normalization
        x_lob_flat = self._res_x_lob.reshape(-1, N_CHANNELS * N_LEVELS)
        self._lob_mean = x_lob_flat.mean(axis=0, dtype=np.float64)
        self._lob_scale = x_lob_flat.std(axis=0, dtype=np.float64)
        self._lob_scale = np.where(self._lob_scale > 1e-6, self._lob_scale, 1.0)
        if self._res_x_glob.shape[-1] > 0:
            self._glob_mean = self._res_x_glob.mean(axis=(0, 1), dtype=np.float64)
            self._glob_scale = self._res_x_glob.std(axis=(0, 1), dtype=np.float64)
            self._glob_scale = np.where(self._glob_scale > 1e-6, self._glob_scale, 1.0)
        else:
            self._glob_mean = np.array([], dtype=np.float64)
            self._glob_scale = np.array([], dtype=np.float64)
        self._y_mean = float(np.mean(self._res_y))
        self._y_std = float(np.std(self._res_y))
        if self._y_std < 1e-8:
            self._y_std = 1.0

        # Standardize reservoir
        lob_norm = self._normalize_lob(self._res_x_lob, fit=False)
        glob_norm = self._normalize_global(self._res_x_glob, fit=False)
        y_norm = (self._res_y - self._y_mean) / self._y_std

        # Train
        n_global = self._res_x_glob.shape[-1] if self._res_x_glob is not None else 0
        self._model = DeepLOBNet(
            n_levels=N_LEVELS, n_channels=N_CHANNELS, n_global=n_global,
            conv_hidden=self.conv_hidden, lstm_hidden=self.lstm_hidden,
            lstm_layers=self.lstm_layers, dropout=self.dropout,
        ).to(self._device)

        opt = torch.optim.AdamW(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()
        n_samples = len(y_norm)

        self._model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n_samples, device="cpu")
            epoch_loss = 0.0
            for i in range(0, n_samples, self.batch_size):
                idx = perm[i:i + self.batch_size]
                xl = torch.from_numpy(lob_norm[idx]).float().to(self._device)
                xg = torch.from_numpy(glob_norm[idx]).float().to(self._device)
                yb = torch.from_numpy(y_norm[idx]).float().to(self._device)

                opt.zero_grad()
                pred = self._model(xl, xg)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * len(idx)
            # Early stopping: if loss stagnates, break
            if epoch > 5 and epoch_loss / n_samples > 0.99:
                # Not improving meaningfully
                pass
        # Free reservoir
        self._res_x_lob = None
        self._res_x_glob = None
        self._res_y = None

    def predict(self, raw):
        n_total = len(raw)
        pred = np.zeros(n_total, dtype=np.float64)
        if self._model is None:
            return pred
        lob, glob, row_ids = self._build_predict_examples(raw)
        if len(row_ids) == 0:
            return pred
        lob_norm = self._normalize_lob(lob, fit=False)
        glob_norm = self._normalize_global(glob, fit=False)

        self._model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(lob_norm), self.batch_size):
                xl = torch.from_numpy(lob_norm[i:i+self.batch_size]).float().to(self._device)
                xg = torch.from_numpy(glob_norm[i:i+self.batch_size]).float().to(self._device)
                p = self._model(xl, xg).cpu().numpy()
                preds.append(p)
        raw_pred = np.concatenate(preds).astype(np.float64)
        # Un-normalize
        raw_pred = raw_pred * self._y_std + self._y_mean
        pred[row_ids] = raw_pred
        return pred

    # -- Internal helpers ---------------------------------------------------

    def _resolve_cols(self, raw):
        if self._active_level_cols is None:
            all_cols = set(raw.columns)
            self._active_level_cols = []
            for chan_cols in LOB_CHANNEL_COLS:
                self._active_level_cols.append([c for c in chan_cols if c in all_cols])
            self._active_global_cols = [c for c in LOB_GLOBAL_COLS if c in all_cols]
        missing = []
        for cols in self._active_level_cols:
            missing.extend([c for c in cols if c not in raw.columns])
        if missing:
            return False
        return True

    def _build_train_examples(self, raw):
        raw = raw.reset_index(drop=True)
        if not self._resolve_cols(raw):
            empty = np.zeros((0, self.lookback, N_CHANNELS, N_LEVELS), dtype=np.float32)
            return empty, np.zeros((0, self.lookback, 0), dtype=np.float32), np.zeros(0, dtype=np.float32), None
        work = raw.loc[:, ["date", "symbol", "interval", "fret12"]].copy()
        work["__row_id"] = np.arange(len(work), dtype=np.int64)
        work = work.sort_values(["date", "symbol", "interval"], kind="mergesort")

        lob_parts, glob_parts, y_parts = [], [], []
        group_cols = ["date", "symbol"]
        for _, gdf in work.groupby(group_cols, sort=False):
            if len(gdf) < self.lookback:
                continue
            row_ids = gdf["__row_id"].to_numpy(dtype=np.int64, copy=False)
            g_idx = raw.iloc[row_ids]

            # Build LOB tensor for this group: (T, C, L)
            T = len(row_ids)
            lob_tensor = np.zeros((T, N_CHANNELS, N_LEVELS), dtype=np.float64)
            for c, col_list in enumerate(self._active_level_cols):
                vals = g_idx.loc[:, col_list].to_numpy(dtype=np.float64, copy=False)
                lob_tensor[:, c, :] = vals

            # Build global features: (T, G)
            if self._active_global_cols:
                glob_feat = g_idx.loc[:, self._active_global_cols].to_numpy(dtype=np.float64, copy=False)
                glob_feat = np.nan_to_num(glob_feat, nan=0.0, posinf=0.0, neginf=0.0)
                glob_feat = np.sign(glob_feat) * np.log1p(np.abs(glob_feat))
            else:
                glob_feat = np.zeros((T, 0), dtype=np.float64)

            # Normalize LOB
            lob_tensor = self._normalize_lob_group(lob_tensor)

            # Sliding windows
            windows_lob = np.lib.stride_tricks.sliding_window_view(
                lob_tensor, (self.lookback, N_CHANNELS, N_LEVELS)
            )[:, 0, 0].copy()  # (n_win, lookback, C, L)

            windows_glob = np.lib.stride_tricks.sliding_window_view(
                glob_feat, (self.lookback, glob_feat.shape[1])
            )[:, 0].copy()  # (n_win, lookback, G)

            targets = gdf["fret12"].to_numpy(dtype=np.float64, copy=False)[self.lookback - 1:]

            lob_parts.append(windows_lob)
            glob_parts.append(windows_glob)
            y_parts.append(targets)

        if not lob_parts:
            return (
                np.zeros((0, self.lookback, N_CHANNELS, N_LEVELS), dtype=np.float32),
                np.zeros((0, self.lookback, 0), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                None,
            )
        return (
            np.concatenate(lob_parts, axis=0).astype(np.float32, copy=False),
            np.concatenate(glob_parts, axis=0).astype(np.float32, copy=False),
            np.concatenate(y_parts, axis=0).astype(np.float32, copy=False),
            None,
        )

    def _build_predict_examples(self, raw):
        raw = raw.reset_index(drop=True)
        if not self._resolve_cols(raw):
            return (np.zeros((0, self.lookback, N_CHANNELS, N_LEVELS), dtype=np.float32),
                    np.zeros((0, self.lookback, 0), dtype=np.float32),
                    np.zeros(0, dtype=np.int64))
        work = raw.loc[:, ["date", "symbol", "interval"]].copy()
        work["__row_id"] = np.arange(len(work), dtype=np.int64)
        work = work.sort_values(["date", "symbol", "interval"], kind="mergesort")

        lob_parts, glob_parts, rid_parts = [], [], []
        group_cols = ["date", "symbol"]
        for _, gdf in work.groupby(group_cols, sort=False):
            if len(gdf) < self.lookback:
                continue
            row_ids = gdf["__row_id"].to_numpy(dtype=np.int64, copy=False)
            g_idx = raw.iloc[row_ids]
            T = len(row_ids)

            lob_tensor = np.zeros((T, N_CHANNELS, N_LEVELS), dtype=np.float64)
            for c, col_list in enumerate(self._active_level_cols):
                vals = g_idx.loc[:, col_list].to_numpy(dtype=np.float64, copy=False)
                lob_tensor[:, c, :] = vals

            if self._active_global_cols:
                glob_feat = g_idx.loc[:, self._active_global_cols].to_numpy(dtype=np.float64, copy=False)
                glob_feat = np.nan_to_num(glob_feat, nan=0.0, posinf=0.0, neginf=0.0)
                glob_feat = np.sign(glob_feat) * np.log1p(np.abs(glob_feat))
            else:
                glob_feat = np.zeros((T, 0), dtype=np.float64)

            lob_tensor = self._normalize_lob_group(lob_tensor)

            windows_lob = np.lib.stride_tricks.sliding_window_view(
                lob_tensor, (self.lookback, N_CHANNELS, N_LEVELS)
            )[:, 0, 0].copy()
            windows_glob = np.lib.stride_tricks.sliding_window_view(
                glob_feat, (self.lookback, glob_feat.shape[1])
            )[:, 0].copy()
            win_rows = row_ids[self.lookback - 1:]

            lob_parts.append(windows_lob)
            glob_parts.append(windows_glob)
            rid_parts.append(win_rows)

        if not lob_parts:
            return (
                np.zeros((0, self.lookback, N_CHANNELS, N_LEVELS), dtype=np.float32),
                np.zeros((0, self.lookback, 0), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
            )
        return (
            np.concatenate(lob_parts, axis=0).astype(np.float32, copy=False),
            np.concatenate(glob_parts, axis=0).astype(np.float32, copy=False),
            np.concatenate(rid_parts, axis=0).astype(np.int64, copy=False),
        )

    def _normalize_lob_group(self, lob_tensor):
        """Per-group LOB normalization: divide prices by midpx, log-transform sizes."""
        lob_tensor = np.nan_to_num(lob_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        # Channel 0, 1 are bid/ask prices; use bid0 as reference (cheap midpx proxy)
        ref = np.maximum(np.abs(lob_tensor[:, 0, 0]), 1e-6)
        ref = ref.reshape(-1, 1, 1)
        lob_tensor[:, 0:2, :] = lob_tensor[:, 0:2, :] / ref - 1.0  # price rel dev
        # Channel 2, 3 are bid/ask sizes; log1p
        lob_tensor[:, 2:4, :] = np.sign(lob_tensor[:, 2:4, :]) * np.log1p(np.abs(lob_tensor[:, 2:4, :]))
        return lob_tensor

    def _normalize_lob(self, x, fit=False):
        """Global z-score normalization across reservoir."""
        B, T, C, L = x.shape
        x_flat = x.reshape(B, T, C * L)
        x_flat = (x_flat - self._lob_mean.reshape(1, 1, -1)) / self._lob_scale.reshape(1, 1, -1)
        return x_flat.reshape(B, T, C, L).astype(np.float32, copy=False)

    def _normalize_global(self, x, fit=False):
        if self._res_x_glob is not None and self._res_x_glob.shape[-1] == 0:
            return np.zeros((x.shape[0], x.shape[1], 0), dtype=np.float32)
        if self._glob_mean is None or len(self._glob_mean) == 0:
            return np.zeros((x.shape[0], x.shape[1], 0), dtype=np.float32)
        return ((x - self._glob_mean.reshape(1, 1, -1)) /
                self._glob_scale.reshape(1, 1, -1)).astype(np.float32, copy=False)
