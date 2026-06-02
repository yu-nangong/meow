"""Shrunk cross-day residual prior conditioned on market state (no label leakage).

Training: uses fret12 to compute true market state (legitimate — labels exist).
Test: uses base forecast CS mean sign as a proxy (no label access).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


class PanelSeasonalityConditionalResidual:
    """Conditional pair prior: market state proxy = sign of CS mean base forecast.

    During training, partial_fit uses ydf["fret12"] for true market state.
    During prediction, predict() uses base_pred for a proxy market state.
    Keys are (symbol, interval, market_state) tuples.
    """

    def __init__(self):
        self.enabled = os.environ.get("MEOW_ENABLE_PANEL_MS", "1") != "0"
        self.blend = float(os.environ.get("MEOW_PANEL_MS_BLEND", "0.12"))
        self.tail_days = int(os.environ.get("MEOW_PANEL_MS_TAIL_DAYS", "40"))
        self.symbol_alpha = float(os.environ.get("MEOW_PANEL_MS_SYMBOL_ALPHA", "20.0"))
        self.interval_alpha = float(os.environ.get("MEOW_PANEL_MS_INTERVAL_ALPHA", "20.0"))
        self.pair_alpha = float(os.environ.get("MEOW_PANEL_MS_PAIR_ALPHA", "40.0"))
        self._global_sum = 0.0
        self._global_count = 0
        self._symbol_sum = {}
        self._symbol_count = {}
        self._interval_sum = {}
        self._interval_count = {}
        self._pair_sum = {}
        self._pair_count = {}
        self._global_mean = 0.0
        self._symbol_mean = {}
        self._interval_mean = {}
        self._pair_mean = {}

    @staticmethod
    def _market_state_from_fret12(ydf: pd.DataFrame) -> np.ndarray:
        """True market state from labels — only used during training."""
        idx = ydf.index.to_frame(index=False)
        grp = pd.DataFrame({
            "date": idx["date"],
            "interval": idx["interval"],
            "fret12": ydf["fret12"].to_numpy(dtype=np.float64, copy=False),
        })
        cs_mean = grp.groupby(["date", "interval"], sort=False)["fret12"].transform("mean")
        raw = cs_mean.to_numpy(dtype=np.float64, copy=False)
        return (raw >= 0.0).astype(np.int32)

    @staticmethod
    def _market_state_from_base_pred(ydf: pd.DataFrame, base_pred: np.ndarray) -> np.ndarray:
        """Proxy market state from base forecast — used during test (no label access)."""
        idx = ydf.index.to_frame(index=False)
        grp = pd.DataFrame({
            "date": idx["date"],
            "interval": idx["interval"],
            "base_pred": np.asarray(base_pred, dtype=np.float64).ravel(),
        })
        cs_mean = grp.groupby(["date", "interval"], sort=False)["base_pred"].transform("mean")
        raw = cs_mean.to_numpy(dtype=np.float64, copy=False)
        return (raw >= 0.0).astype(np.int32)

    def partial_fit(self, ydf: pd.DataFrame, resid: np.ndarray) -> None:
        if not self.enabled or not self.blend or len(ydf) == 0:
            return
        frame = ydf.index.to_frame(index=False).loc[:, ["symbol", "interval"]].copy()
        frame["resid"] = np.asarray(resid, dtype=np.float64)

        self._global_sum += float(frame["resid"].sum())
        self._global_count += int(len(frame))

        symbol_stats = frame.groupby("symbol", sort=False)["resid"].agg(["sum", "count"])
        for symbol, row in symbol_stats.iterrows():
            self._symbol_sum[symbol] = self._symbol_sum.get(symbol, 0.0) + float(row["sum"])
            self._symbol_count[symbol] = self._symbol_count.get(symbol, 0) + int(row["count"])

        interval_stats = frame.groupby("interval", sort=False)["resid"].agg(["sum", "count"])
        for interval, row in interval_stats.iterrows():
            key = int(interval)
            self._interval_sum[key] = self._interval_sum.get(key, 0.0) + float(row["sum"])
            self._interval_count[key] = self._interval_count.get(key, 0) + int(row["count"])

        # During training, use true market state from fret12 (legitimate)
        frame["ms"] = self._market_state_from_fret12(ydf)
        pair_stats = frame.groupby(["symbol", "interval", "ms"], sort=False)["resid"].agg(["sum", "count"])
        for (symbol, interval, ms), row in pair_stats.iterrows():
            key = (symbol, int(interval), int(ms))
            self._pair_sum[key] = self._pair_sum.get(key, 0.0) + float(row["sum"])
            self._pair_count[key] = self._pair_count.get(key, 0) + int(row["count"])

    def finalize_fit(self) -> None:
        if not self._global_count:
            return
        self._global_mean = self._global_sum / self._global_count
        self._symbol_mean = {
            symbol: (self._symbol_sum[symbol] + self.symbol_alpha * self._global_mean)
            / (self._symbol_count[symbol] + self.symbol_alpha)
            for symbol in self._symbol_sum
        }
        self._interval_mean = {
            interval: (self._interval_sum[interval] + self.interval_alpha * self._global_mean)
            / (self._interval_count[interval] + self.interval_alpha)
            for interval in self._interval_sum
        }
        self._pair_mean = {}
        for key, pair_sum in self._pair_sum.items():
            symbol, interval, ms = key
            base_prior = (
                self._symbol_mean.get(symbol, self._global_mean)
                + self._interval_mean.get(interval, self._global_mean)
                - self._global_mean
            )
            pair_count = self._pair_count[key]
            self._pair_mean[key] = (pair_sum + self.pair_alpha * base_prior) / (pair_count + self.pair_alpha)

    def predict(self, ydf: pd.DataFrame, base_pred: np.ndarray = None) -> np.ndarray:
        """Predict using proxy market state from base forecast — no label access."""
        if not self.enabled or not self.blend or len(ydf) == 0:
            return np.zeros(len(ydf), dtype=np.float64)
        assert "fret12" not in ydf.columns.tolist(), "predict() must not read test labels (fret12)"
        if base_pred is None:
            return np.zeros(len(ydf), dtype=np.float64)
        keys = ydf.index.to_frame(index=False).loc[:, ["symbol", "interval"]]
        ms = self._market_state_from_base_pred(ydf, base_pred)
        pred = np.empty(len(keys), dtype=np.float64)
        for idx, row in enumerate(keys.itertuples(index=False)):
            symbol = row.symbol
            interval = int(row.interval)
            base_prior = (
                self._symbol_mean.get(symbol, self._global_mean)
                + self._interval_mean.get(interval, self._global_mean)
                - self._global_mean
            )
            pred[idx] = self._pair_mean.get((symbol, interval, int(ms[idx])), base_prior)
        return self.blend * pred
