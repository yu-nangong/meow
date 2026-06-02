"""Shrunk cross-day residual prior conditioned on market direction and magnitude."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


class PanelSeasonalityMarketResidual:
    """Conditional pair prior: market state = direction + magnitude of CS mean fret12.

    Three states: down (cs mean < -threshold), flat (|cs mean| <= threshold), up (cs mean > threshold).
    Triple the pair keys, adding both direction and magnitude dimensions.
    """

    def __init__(self):
        self.enabled = os.environ.get("MEOW_ENABLE_PANEL_MS", "1") != "0"
        self.blend = float(os.environ.get("MEOW_PANEL_MS_BLEND", "0.12"))
        self.tail_days = int(os.environ.get("MEOW_PANEL_MS_TAIL_DAYS", "40"))
        self.symbol_alpha = float(os.environ.get("MEOW_PANEL_MS_SYMBOL_ALPHA", "20.0"))
        self.interval_alpha = float(os.environ.get("MEOW_PANEL_MS_INTERVAL_ALPHA", "20.0"))
        self.pair_alpha = float(os.environ.get("MEOW_PANEL_MS_PAIR_ALPHA", "40.0"))
        self.ms_threshold_k = float(os.environ.get("MEOW_PANEL_MS_THRESHOLD_K", "0.5"))
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
        self._cs_mean_std = 1e-8

    @staticmethod
    def _cs_mean_values(ydf: pd.DataFrame) -> np.ndarray:
        idx = ydf.index.to_frame(index=False)
        grp = pd.DataFrame({
            "date": idx["date"],
            "interval": idx["interval"],
            "fret12": ydf["fret12"].to_numpy(dtype=np.float64, copy=False),
        })
        cs_mean = grp.groupby(["date", "interval"], sort=False)["fret12"].transform("mean")
        return cs_mean.to_numpy(dtype=np.float64, copy=False)

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

        cs_mean_raw = self._cs_mean_values(ydf)
        chunk_std = float(np.std(cs_mean_raw[np.isfinite(cs_mean_raw)]))
        if chunk_std > 0:
            alpha = 0.1  # smoothing for threshold stability across chunks
            self._cs_mean_std = (1 - alpha) * self._cs_mean_std + alpha * chunk_std
        threshold = self.ms_threshold_k * self._cs_mean_std
        # 0=down, 1=flat, 2=up
        frame["ms"] = np.where(cs_mean_raw < -threshold, 0, np.where(cs_mean_raw > threshold, 2, 1)).astype(np.int32)
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

    def predict(self, ydf: pd.DataFrame) -> np.ndarray:
        if not self.enabled or not self.blend or len(ydf) == 0:
            return np.zeros(len(ydf), dtype=np.float64)
        keys = ydf.index.to_frame(index=False).loc[:, ["symbol", "interval"]]
        cs_mean_raw = self._cs_mean_values(ydf)
        threshold = self.ms_threshold_k * self._cs_mean_std
        ms = np.where(cs_mean_raw < -threshold, 0, np.where(cs_mean_raw > threshold, 2, 1)).astype(np.int32)
        pred = np.empty(len(keys), dtype=np.float64)
        for idx, row in enumerate(keys.itertuples(index=False)):
            symbol = row.symbol
            interval = int(row.interval)
            pair_key = (symbol, interval, int(ms[idx]))
            pred[idx] = self._pair_mean.get(
                pair_key,
                self._symbol_mean.get(symbol, self._global_mean)
                + self._interval_mean.get(interval, self._global_mean)
                - self._global_mean,
            )
        return self.blend * pred
