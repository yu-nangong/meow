from __future__ import annotations

import os

import numpy as np


class IntervalResidualRidge:
    def __init__(self):
        self.enabled = os.environ.get("MEOW_ENABLE_INTERVAL_RESIDUAL", "1") != "0"
        self.alpha = float(os.environ.get("MEOW_INTERVAL_RESIDUAL_ALPHA", "0.5"))
        self.prior_alpha = float(os.environ.get("MEOW_INTERVAL_RESIDUAL_PRIOR_ALPHA", "0.5"))
        self.blend = float(os.environ.get("MEOW_INTERVAL_RESIDUAL_BLEND", "0.4"))
        self.blend_scale = float(os.environ.get("MEOW_INTERVAL_RESIDUAL_BLEND_SCALE", "0.5"))
        self.blend_max_mult = float(os.environ.get("MEOW_INTERVAL_RESIDUAL_BLEND_MAX_MULT", "2.0"))
        self.blend_min_mult = float(os.environ.get("MEOW_INTERVAL_RESIDUAL_BLEND_MIN_MULT", "0.4"))
        self.neighbor_alpha = float(os.environ.get("MEOW_INTERVAL_RESIDUAL_NEIGHBOR_ALPHA", "0.5"))
        self.use_base_pred_rank = os.environ.get("MEOW_INTERVAL_RESIDUAL_USE_BASE_PRED_RANK", "1") != "0"
        self.use_base_pred_rank_tail = (
            os.environ.get("MEOW_INTERVAL_RESIDUAL_USE_BASE_PRED_RANK_TAIL", "1") != "0"
        )
        self.use_base_pred_rank_split_tails = (
            os.environ.get("MEOW_INTERVAL_RESIDUAL_USE_BASE_PRED_RANK_SPLIT_TAILS", "1") != "0"
        )
        self.base_pred_rank_tail_threshold = float(
            os.environ.get("MEOW_INTERVAL_RESIDUAL_BASE_PRED_RANK_TAIL_THRESHOLD", "0.18")
        )
        raw_base_rank_interactions = os.environ.get(
            "MEOW_INTERVAL_RESIDUAL_BASE_RANK_INTERACTIONS",
            "trade_imb_rank_cs,flow_imb_rank_cs,high_minus_low_rank_cs,midpx_level_rank_cs,lastpx_level_rank_cs,tradeBuyQty_level_rank_cs,tradeSellQty_level_rank_cs,bsize0_level_rank_cs,asize0_level_rank_cs,buyVwad_level_rank_cs,sellVwad_level_rank_cs,nTradeBuy_level_rank_cs,nTradeSell_level_rank_cs",
        )
        self.base_rank_interaction_mode = os.environ.get(
            "MEOW_INTERVAL_RESIDUAL_BASE_RANK_INTERACTION_MODE",
            "split_tail",
        ).strip().lower()
        raw_features = os.environ.get(
            "MEOW_INTERVAL_RESIDUAL_FEATURES",
            "trade_imb_rank_cs,flow_imb_rank_cs,micro_dev_rank_cs,ret1_rank_cs,ret3_rank_cs,ret6_rank_cs,ret12_resid_rank_cs,turnover_imb_rank_cs,add_turn_imb_rank_cs,day_open_gap_rank_cs,trade_count_share_rank_cs,buy_vwad_dev_rank_cs,trade_vwad_gap_rank_cs,range_pos_rank_cs,last_mid_dev_rank_cs,ret24_rank_cs,ob_imb4_rank_cs,depth_pressure_04_rank_cs,ob_imb0_rank_cs,ob_imb9_rank_cs,ob_imb19_rank_cs,ob_imb_front_back_rank_cs,ob_imb_inner_outer_rank_cs,spread_rank_cs,ret12_rank_cs,sell_vwad_dev_rank_cs,depth_pressure_59_rank_cs,depth_pressure_1019_rank_cs,depth_pressure_slope_rank_cs,depth_pressure_curve_rank_cs,near_share_imb_rank_cs,top_queue_share_imb_rank_cs,high_gap_rank_cs,low_gap_rank_cs,high_minus_low_rank_cs,trade_buy_high_gap_rank_cs,trade_sell_high_gap_rank_cs,vwad_center_dev_rank_cs,trade_high_center_gap_rank_cs,trade_high_skew_rank_cs,high_vs_trade_high_gap_rank_cs,trade_imb_cs,micro_dev_cs,ret1_cs,ret3_cs,ret6_cs,ret12_cs,ret12_resid_cs,spread_cs,high_gap_cs,trade_buy_high_gap_cs,trade_sell_high_gap_cs,midpx_level_rank_cs,lastpx_level_rank_cs,high_level_rank_cs,low_level_rank_cs,open_level_rank_cs,tradeBuyQty_level_rank_cs,tradeSellQty_level_rank_cs,tradeBuyTurnover_level_rank_cs,tradeSellTurnover_level_rank_cs,buyVwad_level_rank_cs,sellVwad_level_rank_cs,bsize0_level_rank_cs,asize0_level_rank_cs,addBuyQty_level_rank_cs,addSellQty_level_rank_cs,cxlBuyQty_level_rank_cs,cxlSellQty_level_rank_cs,nTradeBuy_level_rank_cs,nTradeSell_level_rank_cs,nAddBuy_level_rank_cs,nAddSell_level_rank_cs,nCxlBuy_level_rank_cs,nCxlSell_level_rank_cs,bid0_level_rank_cs,ask0_level_rank_cs,midpx_zs,lastpx_zs,buyVwad_zs,sellVwad_zs,tradeBuyQty_zs,tradeSellQty_zs,bsize0_zs,asize0_zs,time_sin_2pi,time_cos_2pi,time_sin_4pi,time_cos_4pi",
        )
        self.feature_names = [name.strip() for name in raw_features.split(",") if name.strip()]
        self.base_rank_interaction_features = [
            name.strip() for name in raw_base_rank_interactions.split(",") if name.strip()
        ]
        self._selected_columns = None
        self._selected_base_rank_interactions = None
        self._global_xtx = None
        self._global_xty = None
        self._interval_xtx = None
        self._interval_xty = None
        self._counts = None
        self._global_coef = None
        self._coef = None
        self._interval_blend_weights = None
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
        # compute adaptive per-interval blend weights from training counts
        self._interval_blend_weights = self._compute_blend_weights(n_intervals)
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
            # use per-interval blend weights
            pred[valid] = pred[valid] * self._interval_blend_weights[idx[valid]]
        if np.any(~valid) and self._global_coef is not None:
            pred[~valid] = x[~valid] @ self._global_coef
        # unseen intervals get the global blend
        if np.any(~valid):
            pred[~valid] = pred[~valid] * self.blend
        return pred

    def _select_features(self, xdf, base_pred=None):
        if self._selected_columns is None:
            self._selected_columns = [name for name in self.feature_names if name in xdf.columns]
        if self._selected_base_rank_interactions is None:
            self._selected_base_rank_interactions = [
                name for name in self.base_rank_interaction_features if name in xdf.columns
            ]
        parts = []
        if self._selected_columns:
            parts.append(xdf.loc[:, self._selected_columns].to_numpy(dtype=np.float64, copy=False))
        if self.use_base_pred_rank and base_pred is not None:
            base_rank_centered = self._base_pred_rank_centered(xdf, base_pred)
            parts.append(
                self._base_pred_rank_features(
                    base_rank_centered,
                    include_tail=self.use_base_pred_rank_tail,
                    split_tails=self.use_base_pred_rank_split_tails,
                    tail_threshold=self.base_pred_rank_tail_threshold,
                )
            )
            if self._selected_base_rank_interactions:
                interaction_x = xdf.loc[:, self._selected_base_rank_interactions].to_numpy(dtype=np.float64, copy=False)
                interaction_weights = self._base_pred_rank_interaction_weights(base_rank_centered)
                parts.append(
                    (interaction_x[:, :, None] * interaction_weights[:, None, :]).reshape(len(xdf), -1)
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
    def _base_pred_rank_features(centered, include_tail, split_tails, tail_threshold):
        parts = [centered[:, None]]
        if include_tail:
            if split_tails:
                upper_tail = np.maximum(centered - tail_threshold, 0.0)
                lower_tail = np.maximum(-centered - tail_threshold, 0.0)
                parts.append(upper_tail[:, None])
                parts.append(lower_tail[:, None])
            else:
                tail_excess = np.maximum(np.abs(centered) - tail_threshold, 0.0)
                parts.append((centered * tail_excess)[:, None])
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=1)

    def _base_pred_rank_interaction_weights(self, centered):
        if self.base_rank_interaction_mode == "centered":
            return centered[:, None]
        if self.base_rank_interaction_mode == "tail":
            tail_excess = np.maximum(np.abs(centered) - self.base_pred_rank_tail_threshold, 0.0)
            return (np.sign(centered) * tail_excess)[:, None]
        upper_tail = np.maximum(centered - self.base_pred_rank_tail_threshold, 0.0)
        lower_tail = np.maximum(-centered - self.base_pred_rank_tail_threshold, 0.0)
        return np.column_stack([upper_tail, lower_tail])

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

    def _compute_blend_weights(self, n_intervals):
        """Compute per-interval blend weights proportional to sqrt(count / median count)."""
        counts = self._counts[:n_intervals]
        median_count = float(np.median(counts[counts > 0])) if np.any(counts > 0) else 1.0
        ratio = np.sqrt(counts / np.maximum(median_count, 1.0))
        ratio = np.clip(ratio, self.blend_min_mult, self.blend_max_mult)
        # zero-count intervals get the global blend
        ratio[counts == 0] = 1.0
        return self.blend * ratio
