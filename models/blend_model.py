"""Ensemble blend: (LGB or XGB) + Ridge averaged for complementary signal capture."""
from __future__ import annotations

import os

import numpy as np

from models.lgb_model import LGBModel
from models.xgb_model import XGBModel
from mdl import MeowModel

_TREE_TYPE = os.environ.get("MEOW_TREE_TYPE", "lgb").strip().lower()


class BlendModel:
    """Trains tree model and Ridge in parallel, averages predictions.

    Tree model captures nonlinear interactions; Ridge captures linear structure.
    """

    def __init__(self):
        if _TREE_TYPE == "xgb":
            self._tree = XGBModel()
        else:
            self._tree = LGBModel()
        self._ridge = MeowModel(cacheDir=None)
        self._tree_weight = float(os.environ.get("MEOW_BLEND_TREE_WEIGHT", "0.7"))

    def reset(self):
        self._tree.reset()
        self._ridge.reset()

    def partial_fit(self, xdf, ydf):
        self._tree.partial_fit(xdf, ydf)
        self._ridge.partial_fit(xdf, ydf)

    def finalize_fit(self):
        self._tree.finalize_fit()
        self._ridge.finalize_fit()

    def predict(self, xdf):
        tree_pred = self._tree.predict(xdf)
        ridge_pred = self._ridge.predict(xdf)
        return self._tree_weight * tree_pred + (1.0 - self._tree_weight) * ridge_pred
