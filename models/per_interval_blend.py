"""Per-interval Ridge + global LGB blend.

Trains a separate Ridge model for each time-of-day interval, replacing the
single global Ridge.  Each interval gets its own coefficient vector, capturing
time-of-day-specific linear patterns that a single Ridge can only approximate
via multiplicative time-interaction features.

LGB remains global — tree-based models already partition on time implicitly.
"""
from __future__ import annotations

import os

import numpy as np

from models.lgb_model import LGBModel
from mdl import MeowModel


class PerIntervalBlendModel:
    """Global LGB + per-interval Ridge ensemble.

    Training groups rows by interval and fits independent Ridge accumulators.
    Prediction dispatches to the correct interval's Ridge.
    """

    def __init__(self):
        self._lgb = LGBModel()
        self._lgb_weight = float(os.environ.get("MEOW_BLEND_LGB_WEIGHT", "0.7"))
        self._interval_ridge: dict[int, MeowModel] = {}

    def reset(self):
        self._lgb.reset()
        self._interval_ridge.clear()

    def partial_fit(self, xdf, ydf):
        # LGB sees all rows globally (same as BlendModel)
        self._lgb.partial_fit(xdf, ydf)

        # Per-interval Ridge: group rows by interval, dispatch to interval-specific Ridge
        intervals = xdf.index.get_level_values("interval").to_numpy(copy=False)

        for intv in np.unique(intervals):
            key = int(intv)
            mask = intervals == intv
            if key not in self._interval_ridge:
                ridge = MeowModel(cacheDir=None)
                ridge.reset()
                self._interval_ridge[key] = ridge
            sub_xdf = xdf.loc[mask]
            sub_ydf = ydf.loc[mask]
            self._interval_ridge[key].partial_fit(sub_xdf, sub_ydf)

    def finalize_fit(self):
        self._lgb.finalize_fit()
        for ridge in self._interval_ridge.values():
            ridge.finalize_fit()

    def predict(self, xdf):
        lgb_pred = self._lgb.predict(xdf)

        intervals = xdf.index.get_level_values("interval").to_numpy(copy=False)
        ridge_pred = np.zeros(len(xdf), dtype=np.float64)

        for intv in np.unique(intervals):
            key = int(intv)
            ridge = self._interval_ridge.get(key)
            if ridge is None or not hasattr(ridge, '_coef') or ridge._coef is None:
                continue
            mask = intervals == intv
            sub_xdf = xdf.loc[mask]
            ridge_pred[mask] = ridge.predict(sub_xdf)

        return self._lgb_weight * lgb_pred + (1.0 - self._lgb_weight) * ridge_pred
