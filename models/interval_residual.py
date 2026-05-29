from __future__ import annotations

import os

import numpy as np


class IntervalResidualRidge:
    def __init__(self):
        self.enabled = os.environ.get("MEOW_ENABLE_INTERVAL_RESIDUAL", "1") != "0"
        self.alpha = float(os.environ.get("MEOW_INTERVAL_RESIDUAL_ALPHA", "0.05"))
        self.blend = float(os.environ.get("MEOW_INTERVAL_RESIDUAL_BLEND", "0.15"))
        raw_features = os.environ.get(
            "MEOW_INTERVAL_RESIDUAL_FEATURES",
            ",".join(
                [
                    "trade_imb_rank_cs",
                    "high_gap_rank_cs",
                    "trade_vwad_gap_rank_cs",
                    "high_minus_low_rank_cs",
                    "low_gap_rank_cs",
                    "ret12_resid_rank_cs",
                    "depth_pressure_slope_rank_cs",
                    "top_queue_share_imb_rank_cs",
                ]
            ),
        )
        self.feature_names = [name.strip() for name in raw_features.split(",") if name.strip()]
        self._XtX = None
        self._Xty = None
        self._coef = None
        self._counts = None
        self._selected_columns = None

    def partial_fit(self, xdf, resid):
        if not self.enabled or not self.blend or len(xdf) == 0:
            return
        x = self._select_features(xdf)
        if x.shape[1] == 0:
            return
        intervals = xdf.index.get_level_values("interval").to_numpy(dtype=np.int32, copy=False)
        self._ensure_capacity(int(intervals.max()) + 1, x.shape[1])
        resid = np.asarray(resid, dtype=np.float64).ravel()
        for interval in np.unique(intervals):
            mask = intervals == interval
            xi = x[mask]
            ri = resid[mask]
            self._XtX[interval] += xi.T @ xi
            self._Xty[interval] += xi.T @ ri
            self._counts[interval] += len(ri)

    def finalize_fit(self):
        if self._XtX is None:
            return
        n_intervals, n_features = self._Xty.shape
        self._coef = np.zeros((n_intervals, n_features), dtype=np.float64)
        eye = np.eye(n_features, dtype=np.float64)
        for interval in range(n_intervals):
            if self._counts[interval] == 0:
                continue
            self._coef[interval] = np.linalg.solve(self._XtX[interval] + self.alpha * eye, self._Xty[interval])

    def predict(self, xdf):
        if self._coef is None or not self.enabled or not self.blend or len(xdf) == 0:
            return np.zeros(len(xdf), dtype=np.float64)
        x = self._select_features(xdf)
        if x.shape[1] == 0:
            return np.zeros(len(xdf), dtype=np.float64)
        intervals = xdf.index.get_level_values("interval").to_numpy(dtype=np.int32, copy=False)
        pred = np.zeros(len(xdf), dtype=np.float64)
        valid = intervals < self._coef.shape[0]
        if np.any(valid):
            pred[valid] = np.einsum("ij,ij->i", x[valid], self._coef[intervals[valid]], optimize=True)
        return self.blend * pred

    def _select_features(self, xdf):
        if self._selected_columns is None:
            self._selected_columns = [name for name in self.feature_names if name in xdf.columns]
        if not self._selected_columns:
            return np.zeros((len(xdf), 0), dtype=np.float64)
        return xdf.loc[:, self._selected_columns].to_numpy(dtype=np.float64, copy=False)

    def _ensure_capacity(self, n_intervals, n_features):
        if self._XtX is None:
            self._XtX = np.zeros((n_intervals, n_features, n_features), dtype=np.float64)
            self._Xty = np.zeros((n_intervals, n_features), dtype=np.float64)
            self._counts = np.zeros(n_intervals, dtype=np.int64)
            return
        current = self._XtX.shape[0]
        if n_intervals <= current:
            return
        grow = n_intervals - current
        self._XtX = np.concatenate(
            [self._XtX, np.zeros((grow, n_features, n_features), dtype=np.float64)],
            axis=0,
        )
        self._Xty = np.concatenate([self._Xty, np.zeros((grow, n_features), dtype=np.float64)], axis=0)
        self._counts = np.concatenate([self._counts, np.zeros(grow, dtype=np.int64)], axis=0)
