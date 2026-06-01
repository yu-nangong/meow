"""Ensemble blend: LGB + Ridge with interval-conditioned weighting."""
from __future__ import annotations

import os

import numpy as np

from models.lgb_model import LGBModel
from mdl import MeowModel


class BlendModel:
    """Trains LGB and Ridge in parallel, blends with interval-conditioned weights.

    LGB gets more weight at session open/close (nonlinear dynamics dominate);
    Ridge gets more weight mid-day (linear structure dominates).

    Weight(w) = w_mid + w_amplitude * |interval_frac_centered| * 2
    where interval_frac_centered ∈ [-0.5, +0.5].
    """

    def __init__(self):
        self._lgb = LGBModel()
        self._ridge = MeowModel(cacheDir=None)
        self._lgb_weight = float(os.environ.get("MEOW_BLEND_LGB_WEIGHT", "0.5"))
        self._w_mid = float(os.environ.get("MEOW_BLEND_W_MID", "0.45"))
        self._w_amplitude = float(os.environ.get("MEOW_BLEND_W_AMPLITUDE", "0.15"))

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
        if "interval_frac_centered" in xdf.columns:
            t = xdf["interval_frac_centered"].to_numpy(dtype=np.float64)
            w = self._w_mid + self._w_amplitude * np.abs(t) * 2.0
            w = np.clip(w, 0.0, 1.0)
            return w * lgb_pred + (1.0 - w) * ridge_pred
        return self._lgb_weight * lgb_pred + (1.0 - self._lgb_weight) * ridge_pred
