"""Ensemble blend: LGB + Ridge averaged for complementary signal capture."""
from __future__ import annotations

import os

import numpy as np

from models.lgb_model import LGBModel
from models.torch_model import TorchModel
from mdl import MeowModel


class BlendModel:
    """Trains LGB and Ridge in parallel, averages predictions.

    LGB captures nonlinear interactions; Ridge captures linear structure.
    The ensemble should be more robust than either alone.
    """

    def __init__(self):
        self._lgb = LGBModel()
        self._ridge = MeowModel(cacheDir=None)
        self._lgb_weight = float(os.environ.get("MEOW_BLEND_LGB_WEIGHT", "0.5"))
        self._deep_weight = float(os.environ.get("MEOW_BLEND_DEEP_WEIGHT", "0.08"))
        self._deep = TorchModel() if self._deep_weight > 0.0 else None

    def reset(self):
        self._lgb.reset()
        self._ridge.reset()
        if self._deep is not None:
            self._deep.reset()

    def partial_fit(self, xdf, ydf):
        self._lgb.partial_fit(xdf, ydf)
        self._ridge.partial_fit(xdf, ydf)
        if self._deep is not None:
            self._deep.partial_fit(xdf, ydf)

    def finalize_fit(self):
        self._lgb.finalize_fit()
        self._ridge.finalize_fit()
        if self._deep is not None:
            self._deep.finalize_fit()

    def predict(self, xdf):
        lgb_pred = self._lgb.predict(xdf)
        ridge_pred = self._ridge.predict(xdf)
        base_pred = self._lgb_weight * lgb_pred + (1.0 - self._lgb_weight) * ridge_pred
        if self._deep is None:
            return base_pred
        deep_pred = self._deep.predict(xdf)
        return (1.0 - self._deep_weight) * base_pred + self._deep_weight * deep_pred
