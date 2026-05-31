from __future__ import annotations

import os

import numpy as np


class IntervalResidualRidge:
    def __init__(self):
        self.enabled = os.environ.get("MEOW_ENABLE_INTERVAL_RESIDUAL", "1") != "0"
        self.alpha = float(os.environ.get("MEOW_INTERVAL_RESIDUAL_ALPHA", "0.25"))
        self.prior_alpha = float(os.environ.get("MEOW_INTERVAL_RESIDUAL_PRIOR_ALPHA", "2.0"))
        self.blend = float(os.environ.get("MEOW_INTERVAL_RESIDUAL_BLEND", "0.12"))
        self.neighbor_alpha = float(os.environ.get("MEOW_INTERVAL_RESIDUAL_NEIGHBOR_ALPHA", "0.0"))
        self.use_base_pred_rank = os.environ.get("MEOW_INTERVAL_RESIDUAL_USE_BASE_PRED_RANK", "1") != "0"
        self.use_base_pred_rank_tail = (
            os.environ.get("MEOW_INTERVAL_RESIDUAL_USE_BASE_PRED_RANK_TAIL", "1") != "0"
        )
        self.base_pred_rank_tail_threshold = float(
            os.environ.get("MEOW_INTERVAL_RESIDUAL_BASE_PRED_RANK_TAIL_THRESHOLD", "0.18")
        )
        raw_features = os.environ.get(
            "MEOW_INTERVAL_RESIDUAL_FEATURES",
            ",".join(
                [
                    "trade_imb_rank_cs",
                    "flow_imb_rank_cs",
                    "high_gap_rank_cs",
                    "trade_vwad_gap_rank_cs",
                    "high_minus_low_rank_cs",
                    "low_gap_rank_cs",
                    "ret12_resid_rank_cs",
                    "top_queue_share_imb_rank_cs",
                    "depth_pressure_slope_rank_cs",
                    "ob_imb19_rank_cs",
                ]
            ),
        )
        self.feature_names = [name.strip() for name in raw_features.split(",") if name.strip()]
        self._selected_columns = None
        self._global_xtx = None
        self._global_xty = None
        self._interval_xtx = None
        self._interval_xty = None
        self._counts = None
        self._global_coef = None
        self._coef = None
        self._interval_to_idx = {}

    def partial_fit(self, xdf, resid, base_pred=None):
        if not self.enabled or not self.blend or len(xdf) == 0:
            return
        x = self._select_features(xdf, base_pred=base_pred)
        if x.shape[1] == 0:
            return
        resid = np.asarray(resid, dtype=np.float64).ravel()
        interval_codes = self._interval_codes(xdf.index.get_level_values("interval").to_numpy(copy=False), x.shape[1])
        self._global_xtx += x.T @ x
        self._global_xty += x.T @ resid
        for interval in np.unique(interval_codes):
            mask = interval_codes == interval
            xi = x[mask]
            ri = resid[mask]
            self._interval_xtx[interval] += xi.T @ xi
            self._interval_xty[interval] += xi.T @ ri
            self._counts[interval] += len(ri)

    def finalize_fit(self):
        if self._global_xtx is None:
            return
        n_intervals, n_features = self._interval_xty.shape
        eye = np.eye(n_features, dtype=np.float64)
        self._global_coef = np.linalg.solve(self._global_xtx + self.alpha * eye, self._global_xty)
        self._coef = np.zeros((n_intervals, n_features), dtype=np.float64)
        for interval in range(n_intervals):
            if self._counts[interval] == 0:
                self._coef[interval] = self._global_coef
                continue
            lhs = self._interval_xtx[interval] + (self.alpha + self.prior_alpha) * eye
            rhs = self._interval_xty[interval] + self.prior_alpha * self._global_coef
            self._coef[interval] = np.linalg.solve(lhs, rhs)
        if self.neighbor_alpha > 0.0 and n_intervals > 1:
            self._smooth_neighbor_deltas()

    def predict(self, xdf, base_pred=None):
        if self._coef is None or not self.enabled or not self.blend or len(xdf) == 0:
            return np.zeros(len(xdf), dtype=np.float64)
        x = self._select_features(xdf, base_pred=base_pred)
        if x.shape[1] == 0:
            return np.zeros(len(xdf), dtype=np.float64)
        intervals = xdf.index.get_level_values("interval").to_numpy(copy=False)
        pred = np.zeros(len(xdf), dtype=np.float64)
        idx = np.array([self._interval_to_idx.get(int(interval), -1) for interval in intervals], dtype=np.int32)
        valid = idx >= 0
        if np.any(valid):
            pred[valid] = np.einsum("ij,ij->i", x[valid], self._coef[idx[valid]], optimize=True)
        if np.any(~valid) and self._global_coef is not None:
            pred[~valid] = x[~valid] @ self._global_coef
        return self.blend * pred

    def _select_features(self, xdf, base_pred=None):
        if self._selected_columns is None:
            self._selected_columns = [name for name in self.feature_names if name in xdf.columns]
        parts = []
        if self._selected_columns:
            parts.append(xdf.loc[:, self._selected_columns].to_numpy(dtype=np.float64, copy=False))
        if self.use_base_pred_rank and base_pred is not None:
            base_rank_centered = self._base_pred_rank_centered(xdf, base_pred)
            parts.append(
                self._base_pred_rank_features(
                    base_rank_centered,
                    include_tail=self.use_base_pred_rank_tail,
                    tail_threshold=self.base_pred_rank_tail_threshold,
                )
            )
        if not parts:
            return np.zeros((len(xdf), 0), dtype=np.float64)
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=1)

    def _ensure_capacity(self, n_intervals, n_features):
        if self._global_xtx is None:
            self._global_xtx = np.zeros((n_features, n_features), dtype=np.float64)
            self._global_xty = np.zeros(n_features, dtype=np.float64)
            self._interval_xtx = np.zeros((n_intervals, n_features, n_features), dtype=np.float64)
            self._interval_xty = np.zeros((n_intervals, n_features), dtype=np.float64)
            self._counts = np.zeros(n_intervals, dtype=np.int64)
            return
        current = self._interval_xtx.shape[0]
        if n_intervals <= current:
            return
        grow = n_intervals - current
        self._interval_xtx = np.concatenate(
            [self._interval_xtx, np.zeros((grow, n_features, n_features), dtype=np.float64)],
            axis=0,
        )
        self._interval_xty = np.concatenate(
            [self._interval_xty, np.zeros((grow, n_features), dtype=np.float64)],
            axis=0,
        )
        self._counts = np.concatenate([self._counts, np.zeros(grow, dtype=np.int64)], axis=0)

    def _interval_codes(self, intervals, n_features):
        codes = np.empty(len(intervals), dtype=np.int32)
        next_idx = len(self._interval_to_idx)
        for i, raw_interval in enumerate(intervals):
            key = int(raw_interval)
            idx = self._interval_to_idx.get(key)
            if idx is None:
                idx = next_idx
                self._interval_to_idx[key] = idx
                next_idx += 1
            codes[i] = idx
        self._ensure_capacity(next_idx, n_features)
        return codes

    @staticmethod
    def _base_pred_rank_centered(xdf, base_pred):
        frame = xdf.index.to_frame(index=False)
        frame["base_pred"] = np.asarray(base_pred, dtype=np.float64).ravel()
        ranked = frame.groupby(["date", "interval"], sort=False)["base_pred"].rank(method="average", pct=True)
        return ranked.to_numpy(dtype=np.float64, copy=False) - 0.5

    @staticmethod
    def _base_pred_rank_features(centered, include_tail, tail_threshold):
        parts = [centered[:, None]]
        if include_tail:
            tail_excess = np.maximum(np.abs(centered) - tail_threshold, 0.0)
            parts.append((centered * tail_excess)[:, None])
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=1)

    def _smooth_neighbor_deltas(self):
        deltas = self._coef - self._global_coef[None, :]
        smoothed = deltas.copy()
        for interval in range(len(deltas)):
            accum = deltas[interval].copy()
            weight = 1.0
            if interval > 0:
                accum += self.neighbor_alpha * deltas[interval - 1]
                weight += self.neighbor_alpha
            if interval + 1 < len(deltas):
                accum += self.neighbor_alpha * deltas[interval + 1]
                weight += self.neighbor_alpha
            smoothed[interval] = accum / weight
        self._coef = self._global_coef[None, :] + smoothed
