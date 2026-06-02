"""
Budget-safe raw LOB HistGradientBoostingRegressor arm.
Operates directly on raw LOB columns (no hand-crafted features),
capturing microstructure patterns that ratio-based features might miss.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from log import log

# All raw numerical columns available in the HDF5 files (excluding metadata).
RAW_NUMERIC_COLS = [
    "midpx", "lastpx", "open", "high", "low",
    "bid0", "ask0", "bid4", "ask4", "bid9", "ask9", "bid19", "ask19",
    "bsize0", "asize0",
    "bsize0_4", "asize0_4", "bsize5_9", "asize5_9", "bsize10_19", "asize10_19",
    "btr0_4", "atr0_4", "btr5_9", "atr5_9", "btr10_19", "atr10_19",
    "nTradeBuy", "tradeBuyQty", "tradeBuyTurnover", "tradeBuyHigh", "tradeBuyLow",
    "buyVwad",
    "nTradeSell", "tradeSellQty", "tradeSellTurnover", "tradeSellHigh", "tradeSellLow",
    "sellVwad",
    "nAddBuy", "addBuyQty", "addBuyTurnover", "addBuyHigh", "addBuyLow",
    "nAddSell", "addSellQty", "addSellTurnover", "addSellHigh", "addSellLow",
    "nCxlBuy", "cxlBuyQty", "cxlBuyTurnover", "cxlBuyHigh", "cxlBuyLow",
    "nCxlSell", "cxlSellQty", "cxlSellTurnover", "cxlSellHigh", "cxlSellLow",
]


def _cross_sectional_zscore(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Z-score normalize within (date, interval) cross-section."""
    available = [c for c in cols if c in df.columns]
    if not available:
        return np.zeros((len(df), 0), dtype=np.float32)
    raw = df[available].copy()
    for c in available:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    grp = raw.groupby([df["date"], df["interval"]], sort=False)
    mean = grp.transform("mean")
    std = grp.transform("std").fillna(0.0).clip(lower=1e-8)
    out = (raw - mean) / std
    out = out.fillna(0.0).clip(-10.0, 10.0)
    return out.to_numpy(dtype=np.float32)


class RawHGBTModel:
    """Trains HGBT directly on cross-sectionally normalized raw LOB columns.

    Uses reservoir sampling to control memory and budget.
    Predicts fret12 directly — orthogonal signal arm to blend with Ridge+LGB.
    """

    def __init__(self):
        self._max_iter = int(os.environ.get("MEOW_RAWHGBT_MAX_ITER", "100"))
        self._max_depth = int(os.environ.get("MEOW_RAWHGBT_MAX_DEPTH", "8"))
        self._learning_rate = float(os.environ.get("MEOW_RAWHGBT_LR", "0.05"))
        self._max_rows = int(os.environ.get("MEOW_RAWHGBT_MAX_ROWS", "200000"))
        self._min_samples_leaf = int(os.environ.get("MEOW_RAWHGBT_MIN_LEAF", "20"))
        self._blend = float(os.environ.get("MEOW_RAWHGBT_BLEND", "0.25"))
        self._enabled = os.environ.get("MEOW_RAWHGBT_ENABLED", "1") != "0"
        self._scaler: StandardScaler | None = None
        self._model: HistGradientBoostingRegressor | None = None
        self._active_cols: list[str] | None = None
        self._x_parts: list[np.ndarray] | None = None
        self._y_parts: list[np.ndarray] | None = None
        self._rng = np.random.RandomState(42)
        self._n_seen = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def blend(self) -> float:
        return self._blend

    def reset(self):
        self._scaler = None
        self._model = None
        self._active_cols = None
        self._x_parts = None
        self._y_parts = None
        self._n_seen = 0

    def partial_fit(self, raw_df: pd.DataFrame, target: np.ndarray):
        """Accumulate raw LOB data with reservoir sampling."""
        if self._active_cols is None:
            self._active_cols = [c for c in RAW_NUMERIC_COLS if c in raw_df.columns]
            if not self._active_cols:
                return
        feats = _cross_sectional_zscore(raw_df, self._active_cols)
        if feats.shape[1] == 0:
            return
        y = np.asarray(target, dtype=np.float32)
        n = len(y)
        if self._x_parts is None:
            take = min(n, self._max_rows)
            self._x_parts = [feats[:take].copy()]
            self._y_parts = [y[:take].copy()]
            self._n_seen = n
            return
        capacity = sum(p.shape[0] for p in self._x_parts)
        if capacity < self._max_rows:
            take = min(n, self._max_rows - capacity)
            self._x_parts.append(feats[:take].copy())
            self._y_parts.append(y[:take].copy())
            start_idx = take
        else:
            start_idx = 0
        for i in range(start_idx, n):
            j = self._rng.randint(0, self._n_seen + i + 1)
            if j < self._max_rows:
                reservoir_idx = j % self._max_rows
                part_idx = 0
                offset = 0
                for p in self._x_parts:
                    if reservoir_idx < offset + p.shape[0]:
                        local_idx = reservoir_idx - offset
                        p[local_idx] = feats[i]
                        self._y_parts[part_idx][local_idx] = y[i]
                        break
                    offset += p.shape[0]
                    part_idx += 1
        self._n_seen += n

    def finalize_fit(self):
        if not self._x_parts or not self._y_parts:
            self._enabled = False
            return
        X = np.concatenate(self._x_parts, axis=0).astype(np.float64)
        y = np.concatenate(self._y_parts, axis=0).astype(np.float64)
        del self._x_parts, self._y_parts

        log.inf("RawHGBT fitting on {} rows x {} cols".format(X.shape[0], X.shape[1]))
        self._model = HistGradientBoostingRegressor(
            max_iter=self._max_iter,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            min_samples_leaf=self._min_samples_leaf,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
        )
        self._model.fit(X, y)
        log.inf("RawHGBT fitted: {} iters".format(self._model.n_iter_))

    def predict(self, raw_df: pd.DataFrame) -> np.ndarray:
        if not self._enabled or self._model is None or self._active_cols is None:
            return np.zeros(len(raw_df), dtype=np.float64)
        feats = _cross_sectional_zscore(raw_df, self._active_cols)
        if feats.shape[1] == 0:
            return np.zeros(len(raw_df), dtype=np.float64)
        pred = self._model.predict(feats.astype(np.float64))
        return (self._blend * pred).astype(np.float64)
