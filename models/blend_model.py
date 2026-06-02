"""Ensemble blend: LGB + random subspace Ridge ensemble averaged for complementary signal capture.

When MEOW_RIDGE_ENSEMBLE > 1 (default 5), uses a random subspace ensemble of Ridge
models instead of a single Ridge, exploiting feature subspace diversity
to reduce variance from collinearity.
"""
from __future__ import annotations

import os

import numpy as np

from models.lgb_model import LGBModel
from models.ensemble_ridge import EnsembleRidgeModel


class BlendModel:
    """Trains LGB and Ridge ensemble in parallel, averages predictions.

    LGB captures nonlinear interactions; Ridge ensemble captures linear structure
    with reduced variance from subspace averaging.
    """

    def __init__(self):
        self._lgb = LGBModel()
        self._ridge = EnsembleRidgeModel()
        self._lgb_weight = float(os.environ.get("MEOW_BLEND_LGB_WEIGHT", "0.5"))

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
        return self._lgb_weight * lgb_pred + (1.0 - self._lgb_weight) * ridge_pred
