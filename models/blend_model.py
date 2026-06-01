"""Ensemble blend: LGB + Ridge averaged for complementary signal capture. Supports stacking."""
from __future__ import annotations

import os

import numpy as np

from models.lgb_model import LGBModel
from mdl import MeowModel


class BlendModel:
    """Trains LGB and Ridge in parallel, blends predictions via stacking or fixed weight.

    LGB captures nonlinear interactions; Ridge captures linear structure.
    When stacking is enabled, fits a Ridge combiner on training predictions.
    """

    def __init__(self):
        self._lgb = LGBModel()
        self._ridge = MeowModel(cacheDir=None)
        self._lgb_weight = float(os.environ.get("MEOW_BLEND_LGB_WEIGHT", "0.5"))
        self._stack_coef = None
        self._stack_intercept = 0.0

    def reset(self):
        self._lgb.reset()
        self._ridge.reset()

    def partial_fit(self, xdf, ydf):
        self._lgb.partial_fit(xdf, ydf)
        self._ridge.partial_fit(xdf, ydf)

    def finalize_fit(self):
        self._lgb.finalize_fit()
        self._ridge.finalize_fit()

    def predict_components(self, xdf):
        """Return (lgb_pred, ridge_pred) arrays separately for stacking."""
        return self._lgb.predict(xdf), self._ridge.predict(xdf)

    def set_stacker(self, coef, intercept):
        """Set stacking coefficients from learned combiner.

        coef: [w_lgb, w_ridge, w_int_lgb, w_int_ridge]
        predict = w_lgb*lgb + w_ridge*ridge + w_int_lgb*int_frac*lgb + w_int_ridge*int_frac*ridge + intercept
        """
        self._stack_coef = np.asarray(coef, dtype=np.float64)
        self._stack_intercept = float(intercept)

    def predict(self, xdf):
        lgb_pred = self._lgb.predict(xdf)
        ridge_pred = self._ridge.predict(xdf)
        if self._stack_coef is not None:
            # Build stacking features: [lgb, ridge, lgb*int_frac, ridge*int_frac]
            interval_max = xdf.index.get_level_values("interval").groupby(
                xdf.index.get_level_values("date"), sort=False
            ).transform("max").clip(lower=1)
            int_frac = xdf.index.get_level_values("interval").to_numpy(dtype=np.float64) / interval_max.to_numpy(
                dtype=np.float64
            ) - 0.5
            stack_pred = self._stack_coef[0] * lgb_pred + self._stack_coef[1] * ridge_pred
            stack_pred += self._stack_coef[2] * int_frac * lgb_pred + self._stack_coef[3] * int_frac * ridge_pred
            return stack_pred + self._stack_intercept
        return self._lgb_weight * lgb_pred + (1.0 - self._lgb_weight) * ridge_pred
