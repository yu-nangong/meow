"""Ensemble blend: RankLGB (lambdarank objective) + Ridge."""
from __future__ import annotations

import os

import numpy as np

from models.rank_lgb_model import RankLGBModel
from mdl import MeowModel


class RankBlendModel:
    """Trains RankLGB (lambdarank) and Ridge in parallel, averages predictions.

    RankLGB optimizes for within-group ranking (aligned with Pearson);
    Ridge captures linear structure. Different algorithm family from the
    incumbent LGB regression blend.
    """

    def __init__(self):
        self._lgb = RankLGBModel()
        self._ridge = MeowModel(cacheDir=None)
        self._lgb_weight = float(os.environ.get("MEOW_BLEND_LGB_WEIGHT", "0.7"))

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
