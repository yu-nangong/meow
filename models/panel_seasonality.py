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
        self.half_life_days = float(os.environ.get("MEOW_PANEL_SEASONALITY_HALF_LIFE_DAYS", "20"))
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
        self._last_max_date = 0

    def _decay_stale(self, day_gap: float) -> None:
        """Apply exponential decay to all accumulated statistics."""
        if day_gap <= 0 or self.half_life_days <= 0:
            return
        factor = float(np.exp(-day_gap / self.half_life_days))
        self._global_sum *= factor
        self._global_count = int(round(self._global_count * factor))
        for d in (self._symbol_sum, self._interval_sum, self._pair_sum):
            for k in list(d.keys()):
                d[k] *= factor
        for d in (self._symbol_count, self._interval_count, self._pair_count):
            for k in list(d.keys()):
                d[k] = max(1, int(round(d[k] * factor)))

    def partial_fit(self, ydf: pd.DataFrame, resid: np.ndarray) -> None:
        if not self.enabled or not self.blend or len(ydf) == 0:
            return
        frame = ydf.index.to_frame(index=False).loc[:, ["symbol", "interval"]].copy()
        dates = ydf.index.get_level_values("date").unique()
        current_max_date = int(dates.max()) if len(dates) > 0 else 0
        if self._last_max_date > 0 and current_max_date > self._last_max_date:
            self._decay_stale(float(current_max_date - self._last_max_date))
        self._last_max_date = max(self._last_max_date, current_max_date)
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

        pair_stats = frame.groupby(["symbol", "interval"], sort=False)["resid"].agg(["sum", "count"])
        for (symbol, interval), row in pair_stats.iterrows():
            key = (symbol, int(interval))
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
            symbol, interval = key
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
        pred = np.empty(len(keys), dtype=np.float64)
        for idx, row in enumerate(keys.itertuples(index=False)):
            symbol = row.symbol
            interval = int(row.interval)
            base_prior = (
                self._symbol_mean.get(symbol, self._global_mean)
                + self._interval_mean.get(interval, self._global_mean)
                - self._global_mean
            )
            pred[idx] = self._pair_mean.get((symbol, interval), base_prior)
        return self.blend * pred
