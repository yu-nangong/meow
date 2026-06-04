"""Ensemble blend: three-way XGB + LGB + Ridge for complementary signal capture."""
from __future__ import annotations

import os

import numpy as np

from models.lgb_model import LGBModel
from models.xgb_model import XGBModel
from mdl import MeowModel

_BLEND_MODE = os.environ.get("MEOW_BLEND_MODE", "three_way").strip().lower()


class BlendModel:
    """Three-way blend: XGB rank:pairwise + LGB + Ridge.

    Each model captures different signal:
    - XGB: within-group ranking (rank:pairwise objective)
    - LGB: nonlinear interactions via MSE
    - Ridge: linear structure
    """

    def __init__(self):
        if _BLEND_MODE == "xgb_only":
            self._xgb = XGBModel()
            self._lgb = None
        elif _BLEND_MODE == "lgb_only":
            self._xgb = None
            self._lgb = LGBModel()
        else:  # three_way
            self._xgb = XGBModel()
            self._lgb = LGBModel()
        self._ridge = MeowModel(cacheDir=None)
        self._xgb_weight = float(os.environ.get("MEOW_BLEND_XGB_WEIGHT", "0.4"))
        self._lgb_weight = float(os.environ.get("MEOW_BLEND_LGB_WEIGHT", "0.3"))
        self._ridge_weight = float(os.environ.get("MEOW_BLEND_RIDGE_WEIGHT", "0.3"))

    def reset(self):
        if self._xgb is not None:
            self._xgb.reset()
        if self._lgb is not None:
            self._lgb.reset()
        self._ridge.reset()

    def partial_fit(self, xdf, ydf):
        if self._xgb is not None:
            self._xgb.partial_fit(xdf, ydf)
        if self._lgb is not None:
            self._lgb.partial_fit(xdf, ydf)
        self._ridge.partial_fit(xdf, ydf)

    def finalize_fit(self):
        if self._xgb is not None:
            self._xgb.finalize_fit()
        if self._lgb is not None:
            self._lgb.finalize_fit()
        self._ridge.finalize_fit()

    def predict(self, xdf):
        ridge_pred = self._ridge.predict(xdf)
        pred = self._ridge_weight * ridge_pred
        total_w = self._ridge_weight
        if self._xgb is not None:
            xgb_pred = self._xgb.predict(xdf)
            pred = pred + self._xgb_weight * xgb_pred
            total_w += self._xgb_weight
        if self._lgb is not None:
            lgb_pred = self._lgb.predict(xdf)
            pred = pred + self._lgb_weight * lgb_pred
            total_w += self._lgb_weight
        # Normalize so weights sum to 1
        if total_w > 1e-12:
            pred = pred / total_w
        return pred
