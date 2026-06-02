from __future__ import annotations

import os

import numpy as np
import pandas as pd


class PanelSeasonalityResidual:
    """Shrunk cross-day residual prior on (symbol, interval) keys."""

    def __init__(self):
        self.enabled = os.environ.get("MEOW_ENABLE_PANEL_SEASONALITY", "1") != "0"
        self.blend = float(os.environ.get("MEOW_PANEL_SEASONALITY_BLEND", "0.12"))
        self.tail_days = int(os.environ.get("MEOW_PANEL_SEASONALITY_TAIL_DAYS", "40"))
        self.symbol_alpha = float(os.environ.get("MEOW_PANEL_SEASONALITY_SYMBOL_ALPHA", "20.0"))
        self.interval_alpha = float(os.environ.get("MEOW_PANEL_SEASONALITY_INTERVAL_ALPHA", "20.0"))
        self.pair_alpha = float(os.environ.get("MEOW_PANEL_SEASONALITY_PAIR_ALPHA", "40.0"))
        self.recency_half_life = float(os.environ.get("MEOW_PANEL_SEASONALITY_RECENCY_HALF_LIFE", "20"))
        self.reliability_alpha = float(os.environ.get("MEOW_PANEL_SEASONALITY_RELIABILITY_ALPHA", "40.0"))
        self.default_reliability = float(os.environ.get("MEOW_PANEL_SEASONALITY_DEFAULT_RELIABILITY", "0.3"))
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
        self._pair_reliability = {}

    def partial_fit(self, ydf: pd.DataFrame, resid: np.ndarray, recency_weight: float = 1.0) -> None:
        if not self.enabled or not self.blend or len(ydf) == 0:
            return
        frame = ydf.index.to_frame(index=False).loc[:, ["symbol", "interval"]].copy()
        frame["resid"] = np.asarray(resid, dtype=np.float64)

        self._global_sum += recency_weight * float(frame["resid"].sum())
        self._global_count += recency_weight * int(len(frame))

        symbol_stats = frame.groupby("symbol", sort=False)["resid"].agg(["sum", "count"])
        for symbol, row in symbol_stats.iterrows():
            self._symbol_sum[symbol] = self._symbol_sum.get(symbol, 0.0) + recency_weight * float(row["sum"])
            self._symbol_count[symbol] = self._symbol_count.get(symbol, 0) + recency_weight * int(row["count"])

        interval_stats = frame.groupby("interval", sort=False)["resid"].agg(["sum", "count"])
        for interval, row in interval_stats.iterrows():
            key = int(interval)
            self._interval_sum[key] = self._interval_sum.get(key, 0.0) + recency_weight * float(row["sum"])
            self._interval_count[key] = self._interval_count.get(key, 0) + recency_weight * int(row["count"])

        pair_stats = frame.groupby(["symbol", "interval"], sort=False)["resid"].agg(["sum", "count"])
        for (symbol, interval), row in pair_stats.iterrows():
            key = (symbol, int(interval))
            self._pair_sum[key] = self._pair_sum.get(key, 0.0) + recency_weight * float(row["sum"])
            self._pair_count[key] = self._pair_count.get(key, 0) + recency_weight * int(row["count"])

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
            symbol, interval = key
            base_prior = (
                self._symbol_mean.get(symbol, self._global_mean)
                + self._interval_mean.get(interval, self._global_mean)
                - self._global_mean
            )
            pair_count = self._pair_count[key]
            self._pair_mean[key] = (pair_sum + self.pair_alpha * base_prior) / (pair_count + self.pair_alpha)
        # Per-pair reliability: well-observed pairs get near-full blend; sparse ones get attenuated.
        self._pair_reliability = {
            key: count / (count + self.reliability_alpha)
            for key, count in self._pair_count.items()
        }

    def predict(self, ydf: pd.DataFrame) -> np.ndarray:
        if not self.enabled or not self.blend or len(ydf) == 0:
            return np.zeros(len(ydf), dtype=np.float64)
        keys = ydf.index.to_frame(index=False).loc[:, ["symbol", "interval"]]
        pred = np.empty(len(keys), dtype=np.float64)
        for idx, row in enumerate(keys.itertuples(index=False)):
            symbol = row.symbol
            interval = int(row.interval)
            pair_key = (symbol, interval)
            base_prior = (
                self._symbol_mean.get(symbol, self._global_mean)
                + self._interval_mean.get(interval, self._global_mean)
                - self._global_mean
            )
            pred[idx] = self._pair_mean.get(pair_key, base_prior) * self._pair_reliability.get(pair_key, self.default_reliability)
        return self.blend * pred
