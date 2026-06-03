"""Ensemble blend: LGB + HGBT + Ridge for complementary signal capture."""
from __future__ import annotations

import os

import numpy as np

from models.lgb_model import LGBModel
from models.hgbt_model import HGBTModel
from mdl import MeowModel


class BlendModel:
    """Trains LGB, HGBT, and Ridge in parallel, averages predictions.

    LGB captures nonlinear interactions; HGBT captures alternative nonlinear
    patterns via histogram-based boosting; Ridge captures linear structure.
    The 3-way ensemble should be more robust than any pair alone.
    """

    def __init__(self):
        self._lgb = LGBModel()
        self._hgbt = HGBTModel()
        self._ridge = MeowModel(cacheDir=None)
        self._lgb_weight = float(os.environ.get("MEOW_BLEND_LGB_WEIGHT", "0.6"))
        self._hgbt_weight = float(os.environ.get("MEOW_BLEND_HGBT_WEIGHT", "0.15"))

    def reset(self):
        self._lgb.reset()
        self._hgbt.reset()
        self._ridge.reset()

    def partial_fit(self, xdf, ydf):
        self._lgb.partial_fit(xdf, ydf)
        self._hgbt.partial_fit(xdf, ydf)
        self._ridge.partial_fit(xdf, ydf)

    def finalize_fit(self):
        self._lgb.finalize_fit()
        self._hgbt.finalize_fit()
        self._ridge.finalize_fit()

    def predict(self, xdf):
        lgb_pred = self._lgb.predict(xdf)
        hgbt_pred = self._hgbt.predict(xdf)
        ridge_pred = self._ridge.predict(xdf)
        ridge_weight = 1.0 - self._lgb_weight - self._hgbt_weight
        return (self._lgb_weight * lgb_pred
                + self._hgbt_weight * hgbt_pred
                + ridge_weight * ridge_pred)
