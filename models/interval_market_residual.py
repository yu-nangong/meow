from __future__ import annotations

import os

import numpy as np
import pandas as pd


class IntervalMarketResidualModel:
    """Predict interval-average residuals from interval-aggregated market state."""

    def __init__(self):
        self.enabled = os.environ.get("MEOW_INTERVAL_MARKET_ENABLE", "1") != "0"
        self.alpha = float(os.environ.get("MEOW_INTERVAL_MARKET_ALPHA", "2.0"))
        self.max_blend = float(os.environ.get("MEOW_INTERVAL_MARKET_MAX_BLEND", "1.0"))
        raw_features = os.environ.get(
            "MEOW_INTERVAL_MARKET_FEATURES",
            "trade_imb,turnover_imb,flow_imb,spread,micro_dev,last_mid_dev,"
            "ret1,ret3,ret6,ob_imb0,ob_imb9,ob_imb19,depth_pressure_04,"
            "depth_pressure_59,depth_pressure_1019,buy_vwad_dev,sell_vwad_dev,"
            "trade_vwad_gap,high_gap,low_gap,high_minus_low,range_pos,"
            "trade_count_share",
        )
        self.feature_names = [name.strip() for name in raw_features.split(",") if name.strip()]
        self._selected_columns = None
        self._x_parts = []
        self._y_parts = []
        self._w_parts = []
        self._mean = None
        self._scale = None
        self._coef = None
        self._blend = 0.0

    def reset(self):
        self._selected_columns = None
        self._x_parts = []
        self._y_parts = []
        self._w_parts = []
        self._mean = None
        self._scale = None
        self._coef = None
        self._blend = 0.0

    def partial_fit(self, xdf, resid, base_pred):
        if not self.enabled or len(xdf) == 0:
            return
        x_group, y_group, w_group = self._group_matrix(xdf, resid, base_pred)
        if len(y_group) == 0:
            return
        self._x_parts.append(x_group)
        self._y_parts.append(y_group)
        self._w_parts.append(w_group)

    def finalize_fit(self):
        if not self.enabled or not self._x_parts:
            return
        x = np.concatenate(self._x_parts, axis=0)
        y = np.concatenate(self._y_parts, axis=0)
        w = np.concatenate(self._w_parts, axis=0)
        self._x_parts = []
        self._y_parts = []
        self._w_parts = []

        self._mean = x.mean(axis=0)
        self._scale = x.std(axis=0)
        self._scale = np.where(self._scale > 1e-12, self._scale, 1.0)
        z = (x - self._mean) / self._scale
        sqrt_w = np.sqrt(np.maximum(w, 1.0))
        zw = z * sqrt_w[:, None]
        yw = y * sqrt_w
        xtx = zw.T @ zw
        xty = zw.T @ yw
        self._coef = np.linalg.solve(
            xtx + self.alpha * np.eye(z.shape[1], dtype=np.float64),
            xty,
        )
        pred = z @ self._coef
        denom = float(pred @ pred)
        if denom > 1e-12:
            self._blend = float(np.clip((pred @ y) / denom, 0.0, self.max_blend))
        else:
            self._blend = 0.0

    def predict(self, xdf, base_pred):
        if self._coef is None or self._blend == 0.0 or len(xdf) == 0:
            return np.zeros(len(xdf), dtype=np.float64)
        x_group, _, _, group_keys, row_group_id = self._group_matrix(
            xdf, None, base_pred, return_group_id=True
        )
        z = (x_group - self._mean) / self._scale
        group_pred = self._blend * (z @ self._coef)
        group_lookup = {int(key): idx for idx, key in enumerate(group_keys)}
        row_idx = np.array([group_lookup[int(key)] for key in row_group_id], dtype=np.int32)
        return group_pred[row_idx]

    def _group_matrix(self, xdf, resid, base_pred, return_group_id=False):
        if self._selected_columns is None:
            self._selected_columns = [name for name in self.feature_names if name in xdf.columns]
        frame = pd.DataFrame(
            {
                "date": xdf.index.get_level_values("date").to_numpy(copy=False),
                "interval": xdf.index.get_level_values("interval").to_numpy(copy=False),
                "base_pred": np.asarray(base_pred, dtype=np.float64).ravel(),
            }
        )
        if resid is not None:
            frame["resid"] = np.asarray(resid, dtype=np.float64).ravel()
        for name in self._selected_columns:
            frame[name] = xdf[name].to_numpy(dtype=np.float64, copy=False)
        grp = frame.groupby(["date", "interval"], sort=False)

        out = pd.DataFrame(index=grp.size().index)
        out["base_pred_mean"] = grp["base_pred"].mean()
        out["base_pred_median"] = grp["base_pred"].median()
        out["base_pred_std"] = grp["base_pred"].std().fillna(0.0)
        for name in self._selected_columns:
            out[f"{name}_mean"] = grp[name].mean()
            out[f"{name}_std"] = grp[name].std().fillna(0.0)
        out["count"] = grp.size().astype(np.float64)
        intervals = out.index.get_level_values("interval").to_numpy(dtype=np.float64, copy=False)
        max_interval = max(float(np.max(intervals)), 1.0)
        frac = intervals / max_interval - 0.5
        out["interval_frac_centered"] = frac
        out["interval_abs_centered"] = np.abs(frac)

        y_group = (
            grp["resid"].mean().to_numpy(dtype=np.float64, copy=False)
            if resid is not None
            else np.zeros(len(out), dtype=np.float64)
        )
        x_group = out.drop(columns=["count"]).to_numpy(dtype=np.float64, copy=False)
        w_group = out["count"].to_numpy(dtype=np.float64, copy=False)
        if return_group_id:
            group_keys = (
                out.index.get_level_values("date").to_numpy(dtype=np.int64, copy=False) * 1000
                + out.index.get_level_values("interval").to_numpy(dtype=np.int64, copy=False)
            )
            row_group_id = (
                frame["date"].to_numpy(dtype=np.int64, copy=False) * 1000
                + frame["interval"].to_numpy(dtype=np.int64, copy=False)
            )
            return x_group, y_group, w_group, group_keys, row_group_id
        return x_group, y_group, w_group
