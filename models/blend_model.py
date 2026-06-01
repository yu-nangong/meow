"""Ensemble blend: LGB + Ridge averaged for complementary signal capture."""
from __future__ import annotations

import os

import numpy as np

from models.lgb_model import LGBModel
from mdl import MeowModel


class BlendModel:
    """Trains LGB and Ridge in parallel, averages predictions.

    Supports interval-conditioned blending: LGB weight varies with
    interval position (higher at open/close, lower mid-day) via a
    U-shaped profile controlled by amplitude and power parameters.

    LGB captures nonlinear interactions; Ridge captures linear structure.
    The ensemble should be more robust than either alone.
    """

    def __init__(self):
        self._lgb = LGBModel()
        self._ridge = MeowModel(cacheDir=None)
        self._lgb_weight = float(os.environ.get("MEOW_BLEND_LGB_WEIGHT", "0.5"))
        self._interval_amplitude = float(os.environ.get("MEOW_BLEND_INTERVAL_AMPLITUDE", "0.15"))
        self._interval_power = float(os.environ.get("MEOW_BLEND_INTERVAL_POWER", "2.0"))

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
        if self._interval_amplitude <= 0 or "interval_frac_centered" not in xdf.columns:
            return self._lgb_weight * lgb_pred + (1.0 - self._lgb_weight) * ridge_pred
        t = np.abs(xdf["interval_frac_centered"].to_numpy(dtype=np.float64)) * 2.0
        offset = self._interval_amplitude * np.power(t, self._interval_power)
        w_lgb = np.clip(self._lgb_weight + offset, 0.0, 1.0)
        return w_lgb * lgb_pred + (1.0 - w_lgb) * ridge_pred
