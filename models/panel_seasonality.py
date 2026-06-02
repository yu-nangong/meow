from __future__ import annotations

import os

import numpy as np
import pandas as pd


class PanelSeasonalityResidual:
    """Shrunk cross-day residual prior on (symbol, interval) keys.

    Supports exponential recency weighting via MEOW_PANEL_SEASONALITY_HALF_LIFE.
    When half_life > 0, more recent days get higher weight: weight = exp(-days_back / half_life).
    """

    def __init__(self):
        self.enabled = os.environ.get("MEOW_ENABLE_PANEL_SEASONALITY", "1") != "0"
        self.blend = float(os.environ.get("MEOW_PANEL_SEASONALITY_BLEND", "0.12"))
        self.tail_days = int(os.environ.get("MEOW_PANEL_SEASONALITY_TAIL_DAYS", "40"))
        self.symbol_alpha = float(os.environ.get("MEOW_PANEL_SEASONALITY_SYMBOL_ALPHA", "20.0"))
        self.interval_alpha = float(os.environ.get("MEOW_PANEL_SEASONALITY_INTERVAL_ALPHA", "20.0"))
        self.half_life = float(os.environ.get("MEOW_PANEL_SEASONALITY_HALF_LIFE", "0.0"))
        self.pair_alpha = float(os.environ.get("MEOW_PANEL_SEASONALITY_PAIR_ALPHA", "40.0"))
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
        self._ref_date = int(os.environ.get("MEOW_PANEL_SEASONALITY_REF_DATE", "0"))

    def _row_weights(self, ydf: pd.DataFrame) -> np.ndarray | None:
        """Return per-row recency weights based on days since reference date."""
        if self.half_life <= 0.0:
            return None
        dates = ydf.index.get_level_values("date").to_numpy(copy=False).astype(np.int32)
        if self._ref_date == 0:
            self._ref_date = int(dates.max())
        days_back = np.maximum(self._ref_date - dates, 0).astype(np.float64)
        return np.exp(-days_back / self.half_life)

    def partial_fit(self, ydf: pd.DataFrame, resid: np.ndarray) -> None:
        if not self.enabled or not self.blend or len(ydf) == 0:
            return
        resid = np.asarray(resid, dtype=np.float64)
        weights = self._row_weights(ydf)

        if weights is not None:
            w = weights
            self._global_sum += float(np.dot(w, resid))
            self._global_count += float(w.sum())

            frame = ydf.index.to_frame(index=False).loc[:, ["symbol", "interval"]]
            frame["resid"] = resid
            frame["weight"] = w

            def _wagg(g):
                wg = g["weight"].to_numpy(dtype=np.float64, copy=False)
                rg = g["resid"].to_numpy(dtype=np.float64, copy=False)
                return pd.Series({"sum": float(np.dot(wg, rg)), "count": float(wg.sum())})

            symbol_stats = frame.groupby("symbol", sort=False, group_keys=False).apply(_wagg, include_groups=False)
            interval_stats = frame.groupby("interval", sort=False, group_keys=False).apply(_wagg, include_groups=False)
            pair_stats = frame.groupby(["symbol", "interval"], sort=False, group_keys=False).apply(_wagg, include_groups=False)
        else:
            self._global_sum += float(resid.sum())
            self._global_count += int(len(ydf))

            frame = ydf.index.to_frame(index=False).loc[:, ["symbol", "interval"]]
            frame["resid"] = resid
            symbol_stats = frame.groupby("symbol", sort=False)["resid"].agg(["sum", "count"])
            interval_stats = frame.groupby("interval", sort=False)["resid"].agg(["sum", "count"])
            pair_stats = frame.groupby(["symbol", "interval"], sort=False)["resid"].agg(["sum", "count"])

        for symbol, row in symbol_stats.iterrows():
            self._symbol_sum[symbol] = self._symbol_sum.get(symbol, 0.0) + float(row["sum"])
            self._symbol_count[symbol] = self._symbol_count.get(symbol, 0.0) + float(row["count"])

        for interval, row in interval_stats.iterrows():
            key = int(interval)
            self._interval_sum[key] = self._interval_sum.get(key, 0.0) + float(row["sum"])
            self._interval_count[key] = self._interval_count.get(key, 0.0) + float(row["count"])

        for (symbol, interval), row in pair_stats.iterrows():
            key = (symbol, int(interval))
            self._pair_sum[key] = self._pair_sum.get(key, 0.0) + float(row["sum"])
            self._pair_count[key] = self._pair_count.get(key, 0.0) + float(row["count"])

    def finalize_fit(self) -> None:
        if not self._global_count:
            return
        self._global_mean = self._global_sum / max(self._global_count, 1e-12)
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
