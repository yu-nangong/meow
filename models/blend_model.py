"""Ensemble blend: LGB + Ridge averaged for complementary signal capture."""
from __future__ import annotations

import os

import numpy as np

from models.lgb_model import LGBModel
from mdl import MeowModel


class BlendModel:
    """Trains LGB and Ridge in parallel, averages predictions.

    LGB captures nonlinear interactions; Ridge captures linear structure.
    The ensemble should be more robust than either alone.
    """

    def __init__(self):
        self._lgb = LGBModel()
        self._ridge = MeowModel(cacheDir=None)
        self._lgb_weight = float(os.environ.get("MEOW_BLEND_LGB_WEIGHT", "0.55"))
        self._interval_bias = float(os.environ.get("MEOW_BLEND_INTERVAL_BIAS", "0.05"))

    def reset(self):
        self._lgb.reset()
        self._ridge.reset()

    def partial_fit(self, xdf, ydf):
        self._lgb.partial_fit(xdf, ydf)
        self._ridge.partial_fit(xdf, ydf)

    def finalize_fit(self):
        self._lgb.finalize_fit()
        self._ridge.finalize_fit()

    def predict(self, xdf):
        lgb_pred = self._lgb.predict(xdf)
        ridge_pred = self._ridge.predict(xdf)
        if "interval_frac_centered" in xdf.columns and self._interval_bias != 0.0:
            interval_frac = xdf["interval_frac_centered"].to_numpy(dtype=np.float64)
            w_lgb = self._lgb_weight + self._interval_bias * np.abs(interval_frac)
            w_lgb = np.clip(w_lgb, 0.0, 1.0)
        else:
            w_lgb = self._lgb_weight
        return w_lgb * lgb_pred + (1.0 - w_lgb) * ridge_pred
