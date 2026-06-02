"""Raw LOB neural network: MLP consuming normalized order book price/size data.

Unlike the tabular feature engineering approach (slope, curvature, imbalances),
this model directly consumes raw bid/ask price levels and size distributions,
learning nonlinear order book shape from the microstructure data itself.

Zhang et al., DeepLOB (2019) — adapted: simplified CNN+LSTM -> pure MLP
on single-snapshot normalized LOB shape for residual correction.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


RAW_LOB_COLS = [
    "bid0", "ask0",
    "bid4", "ask4",
    "bid9", "ask9",
    "bid19", "ask19",
    "bsize0", "asize0",
    "bsize0_4", "asize0_4",
    "bsize5_9", "asize5_9",
    "bsize10_19", "asize10_19",
    "btr0_4", "atr0_4",
    "btr5_9", "atr5_9",
    "btr10_19", "atr10_19",
]

N_RAW = len(RAW_LOB_COLS) + 1  # 23: 22 raw cols + log_midpx


def _extract_normalized(raw_df, eps=1e-8):
    """Extract raw LOB columns and normalize to ratios.

    Prices: all divided by midpx (makes cross-stock comparable)
    Sizes: divided by total bid/ask size (proportions)
    Turnover: divided by total turnover
    log_midpx: absolute price-level anchor

    Returns np.float32 array of shape (n_rows, N_RAW).
    """
    midpx = raw_df["midpx"].to_numpy(dtype=np.float64, copy=False)
    midpx_safe = np.maximum(np.abs(midpx), eps)

    # Price ratios (8 cols): bid_i/midpx, ask_i/midpx
    prices = np.column_stack([
        raw_df[c].to_numpy(dtype=np.float64, copy=False) / midpx_safe
        for c in RAW_LOB_COLS[:8]
    ])

    # Size proportions (8 cols): size_i / total_side_size
    bid_total = (
        raw_df["bsize0_4"].to_numpy(dtype=np.float64, copy=False)
        + raw_df["bsize5_9"].to_numpy(dtype=np.float64, copy=False)
        + raw_df["bsize10_19"].to_numpy(dtype=np.float64, copy=False)
    )
    ask_total = (
        raw_df["asize0_4"].to_numpy(dtype=np.float64, copy=False)
        + raw_df["asize5_9"].to_numpy(dtype=np.float64, copy=False)
        + raw_df["asize10_19"].to_numpy(dtype=np.float64, copy=False)
    )
    bid_total_safe = np.maximum(bid_total, eps)
    ask_total_safe = np.maximum(ask_total, eps)

    sizes = np.column_stack([
        raw_df["bsize0"].to_numpy(dtype=np.float64, copy=False) / bid_total_safe,
        raw_df["asize0"].to_numpy(dtype=np.float64, copy=False) / ask_total_safe,
        raw_df["bsize0_4"].to_numpy(dtype=np.float64, copy=False) / bid_total_safe,
        raw_df["asize0_4"].to_numpy(dtype=np.float64, copy=False) / ask_total_safe,
        raw_df["bsize5_9"].to_numpy(dtype=np.float64, copy=False) / bid_total_safe,
        raw_df["asize5_9"].to_numpy(dtype=np.float64, copy=False) / ask_total_safe,
        raw_df["bsize10_19"].to_numpy(dtype=np.float64, copy=False) / bid_total_safe,
        raw_df["asize10_19"].to_numpy(dtype=np.float64, copy=False) / ask_total_safe,
    ])

    # Turnover proportions (6 cols)
    tr_total = (
        raw_df["btr0_4"].to_numpy(dtype=np.float64, copy=False)
        + raw_df["btr5_9"].to_numpy(dtype=np.float64, copy=False)
        + raw_df["btr10_19"].to_numpy(dtype=np.float64, copy=False)
        + raw_df["atr0_4"].to_numpy(dtype=np.float64, copy=False)
        + raw_df["atr5_9"].to_numpy(dtype=np.float64, copy=False)
        + raw_df["atr10_19"].to_numpy(dtype=np.float64, copy=False)
    )
    tr_total_safe = np.maximum(tr_total, eps)
    turnovers = np.column_stack([
        raw_df[c].to_numpy(dtype=np.float64, copy=False) / tr_total_safe
        for c in RAW_LOB_COLS[16:22]
    ])

    # Log midpx (1 col): absolute price-level anchor
    log_midpx = np.log(midpx_safe + eps).reshape(-1, 1)

    # Concatenate all features
    feats = np.column_stack([prices, sizes, turnovers, log_midpx])

    # Replace inf/nan with 0
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats.astype(np.float32)


class RawLobMLP(nn.Module):
    """Simple MLP on normalized raw LOB data.

    Architecture: 23 -> 64 -> 32 -> 1  (3,649 parameters)
    Lightweight enough for CPU chunked SGD training.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_RAW, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RawLobResidualModel:
    """Wrapper: trains RawLobMLP on blend residuals, adds correction at test time.

    Training: chunked SGD with Adam. Each chunk:
      1. extract raw LOB features from raw DataFrame
      2. compute base blend predictions
      3. compute residuals = y_true - base_pred
      4. take SGD steps to minimize MSE(residuals, NN(raw_lob))

    Prediction: NN(raw_lob) -> residual estimate -> add to base blend prediction.
    """

    def __init__(self):
        self._nn = RawLobMLP()
        self._optimizer = None
        self._n_updates = 0
        self._lr = 0.001
        self._batch_size = 4096
        self._n_epochs_per_chunk = 3

    def reset(self):
        self._nn = RawLobMLP()
        self._optimizer = None
        self._n_updates = 0

    def partial_fit(self, raw_df, base_pred, y_true):
        """One chunk of training: extract features, compute residuals, take SGD steps."""
        X = _extract_normalized(raw_df)
        resid = np.asarray(y_true, dtype=np.float32).ravel() - np.asarray(base_pred, dtype=np.float32).ravel()

        # Remove extreme residuals (beyond 5 sigma) for training stability
        resid_std = np.std(resid)
        if resid_std > 0:
            clip = 5.0 * resid_std
            mask = np.abs(resid) < clip
            X = X[mask]
            resid = resid[mask]

        if len(X) < self._batch_size:
            return

        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self._nn.parameters(), lr=self._lr, weight_decay=1e-5)

        n_samples = len(X)
        indices = np.arange(n_samples)

        for _ in range(self._n_epochs_per_chunk):
            np.random.shuffle(indices)
            for start in range(0, n_samples, self._batch_size):
                batch_idx = indices[start:start + self._batch_size]
                xb = torch.from_numpy(X[batch_idx])
                yb = torch.from_numpy(resid[batch_idx])

                self._optimizer.zero_grad()
                pred = self._nn(xb)
                loss = nn.functional.mse_loss(pred, yb)
                loss.backward()
                self._optimizer.step()
                self._n_updates += 1

    def finalize_fit(self):
        """No-op; all training done incrementally."""

    def predict(self, raw_df):
        """Predict residual correction for given raw LOB data."""
        X = _extract_normalized(raw_df)
        with torch.no_grad():
            xb = torch.from_numpy(X)
            return self._nn(xb).numpy().astype(np.float64)

    @property
    def enabled(self):
        return True
