"""
Budget-safe raw LOB residual MLP using sklearn.
Takes 22 normalized raw LOB columns, predicts residual on top of blend forecast.
No torch dependency — sklearn MLPRegressor is lightweight and import-safe.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from log import log

# Raw LOB columns consumed by this model (subset of LOB_PRICE_SIZE_COLS from data_io).
RAW_LOB_COLS = [
    "bid0", "ask0", "bid4", "ask4", "bid9", "ask9", "bid19", "ask19",
    "bsize0", "asize0", "bsize0_4", "asize0_4", "bsize5_9", "asize5_9",
    "bsize10_19", "asize10_19",
    "btr0_4", "atr0_4", "btr5_9", "atr5_9", "btr10_19", "atr10_19",
]


def _normalize_raw_lob(df: pd.DataFrame) -> np.ndarray:
    """Cross-sectional z-score normalize raw LOB columns within (date, interval)."""
    available = [c for c in RAW_LOB_COLS if c in df.columns]
    if not available:
        return np.zeros((len(df), 0), dtype=np.float32)
    raw = df[available].copy()
    # Guard against inf/nan in raw data
    for c in available:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Cross-sectional normalize within each (date, interval)
    grp = raw.groupby([df["date"], df["interval"]], sort=False)
    mean = grp.transform("mean")
    std = grp.transform("std").fillna(0.0).clip(lower=1e-8)
    out = (raw - mean) / std
    out = out.fillna(0.0).clip(-10.0, 10.0)
    return out.to_numpy(dtype=np.float32)


class RawLOBMLP:
    """Trains a tiny MLP on normalized raw LOB columns to predict residual.

    Architecture: input -> 32 -> 16 -> 1 (ReLU, early stopping).
    Total params ~1K — budget-safe.
    """

    def __init__(self):
        self._hidden = tuple(
            int(x) for x in os.environ.get("MEOW_RAWLOB_HIDDEN", "32,16").split(",")
        )
        self._alpha = float(os.environ.get("MEOW_RAWLOB_ALPHA", "0.001"))
        self._max_iter = int(os.environ.get("MEOW_RAWLOB_MAX_ITER", "200"))
        self._blend = float(os.environ.get("MEOW_RAWLOB_BLEND", "0.3"))
        self._scaler: StandardScaler | None = None
        self._mlp: MLPRegressor | None = None
        self._enabled = os.environ.get("MEOW_RAWLOB_ENABLED", "1") != "0"

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def blend(self) -> float:
        return self._blend

    def reset(self):
        self._scaler = None
        self._mlp = None

    def partial_fit(self, raw_df: pd.DataFrame, residual: np.ndarray):
        """Accumulate raw LOB data and residuals for final fit."""
        feats = _normalize_raw_lob(raw_df)
        if feats.shape[1] == 0:
            return
        if not hasattr(self, "_x_parts"):
            self._x_parts = []
            self._y_parts = []
        self._x_parts.append(feats)
        self._y_parts.append(np.asarray(residual, dtype=np.float32))

    def finalize_fit(self):
        if not hasattr(self, "_x_parts") or not self._x_parts:
            self._enabled = False
            return
        X = np.concatenate(self._x_parts, axis=0).astype(np.float64)
        y = np.concatenate(self._y_parts, axis=0).astype(np.float64)
        del self._x_parts, self._y_parts

        # Sub-sample to at most 500K rows for budget safety
        n = len(X)
        if n > 500_000:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, size=500_000, replace=False)
            X, y = X[idx], y[idx]

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        del X

        self._mlp = MLPRegressor(
            hidden_layer_sizes=self._hidden,
            activation="relu",
            solver="adam",
            alpha=self._alpha,
            batch_size=4096,
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=self._max_iter,
            tol=1e-5,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
        )
        self._mlp.fit(X_scaled, y)
        n_iter = self._mlp.n_iter_ if hasattr(self._mlp, "n_iter_") else self._max_iter
        log.inf(
            "RawLOB MLP fitted: {} iters, final loss={:.6f}".format(
                n_iter, self._mlp.loss_ if hasattr(self._mlp, "loss_") else float("nan")
            )
        )

    def predict(self, raw_df: pd.DataFrame) -> np.ndarray:
        if not self._enabled or self._mlp is None:
            return np.zeros(len(raw_df), dtype=np.float32)
        feats = _normalize_raw_lob(raw_df)
        if feats.shape[1] == 0:
            return np.zeros(len(raw_df), dtype=np.float32)
        X = feats.astype(np.float64)
        if self._scaler is not None:
            X = self._scaler.transform(X)
        pred = self._mlp.predict(X)
        return (self._blend * pred).astype(np.float64)
