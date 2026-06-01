"""Ensemble blend: Ridge plus a configurable nonlinear arm."""
from __future__ import annotations

import os

import numpy as np

from models.lgb_model import LGBModel
from mdl import MeowModel


class BlendModel:
    """Train Ridge with LightGBM, FT-style attention, or a BatchEnsemble MLP arm."""

    def __init__(self):
        nonlinear_model = os.environ.get("MEOW_BLEND_NONLINEAR_MODEL", "tabm").strip().lower()
        if nonlinear_model == "torch":
            from models.torch_model import TorchModel

            self._nonlinear = TorchModel()
            default_weight = "0.35"
        elif nonlinear_model == "tabm":
            from models.tabm_model import TabMModel

            self._nonlinear = TabMModel()
            default_weight = "0.30"
        else:
            self._nonlinear = LGBModel()
            default_weight = os.environ.get("MEOW_BLEND_LGB_WEIGHT", "0.5")
        self._ridge = MeowModel(cacheDir=None)
        self._nonlinear_weight = float(os.environ.get("MEOW_BLEND_NONLINEAR_WEIGHT", default_weight))

    def reset(self):
        self._nonlinear.reset()
        self._ridge.reset()

    def partial_fit(self, xdf, ydf):
        self._nonlinear.partial_fit(xdf, ydf)
        self._ridge.partial_fit(xdf, ydf)

    def finalize_fit(self):
        self._nonlinear.finalize_fit()
        self._ridge.finalize_fit()

    def predict(self, xdf):
        nonlinear_pred = self._nonlinear.predict(xdf)
        ridge_pred = self._ridge.predict(xdf)
        return self._nonlinear_weight * nonlinear_pred + (1.0 - self._nonlinear_weight) * ridge_pred
