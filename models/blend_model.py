"""Ensemble blend: LGB + Ridge averaged for complementary signal capture."""
from __future__ import annotations

import os

import numpy as np

from models.lgb_model import LGBModel
from mdl import MeowModel


class BlendModel:
    """Trains LGB and Ridge in parallel, averages predictions.

    LGB captures nonlinear interactions; Ridge captures linear structure.
    The ensemble should be more robust than either alone.
    """

    def __init__(self):
        self._lgb = LGBModel()
        self._ridge = MeowModel(cacheDir=None)
        self._lgb_weight = float(os.environ.get("MEOW_BLEND_LGB_WEIGHT", "0.7"))
        self._learned_coef = None
        self._per_symbol_biases = None

    def reset(self):
        self._lgb.reset()
        self._ridge.reset()
        self._learned_coef = None
        self._per_symbol_biases = None

    def partial_fit(self, xdf, ydf):
        self._lgb.partial_fit(xdf, ydf)
        self._ridge.partial_fit(xdf, ydf)

    def finalize_fit(self):
        self._lgb.finalize_fit()
        self._ridge.finalize_fit()

    def predict(self, xdf):
        import numpy as np
        lgb_pred = self._lgb.predict(xdf)
        ridge_pred = self._ridge.predict(xdf)
        if self._learned_coef is not None:
            a, b, c = self._learned_coef
            pred = a * lgb_pred + b * ridge_pred + c
        else:
            pred = self._lgb_weight * lgb_pred + (1.0 - self._lgb_weight) * ridge_pred
        if self._per_symbol_biases is not None and hasattr(xdf, 'index'):
            syms = xdf.index.get_level_values("symbol").to_numpy()
            sym_bias = np.array([self._per_symbol_biases.get(s, 0.0) for s in syms], dtype=np.float64)
            nan_mask = ~np.isfinite(sym_bias)
            sym_bias[nan_mask] = 0.0
            pred = pred - sym_bias
        return pred

    def predict_sub(self, xdf):
        """Return (lgb_pred, ridge_pred) for meta-blend learning."""
        return self._lgb.predict(xdf), self._ridge.predict(xdf)

    def set_learned_coef(self, a, b, c):
        self._learned_coef = (float(a), float(b), float(c))

    def set_per_symbol_biases(self, biases):
        self._per_symbol_biases = dict(biases)
