"""Ensemble blend: Ridge fit first, then tree model on Ridge residuals for complementary signal."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from models.lgb_model import LGBModel
from models.xgb_model import XGBModel
from mdl import MeowModel

_TREE_TYPE = os.environ.get("MEOW_TREE_TYPE", "lgb").strip().lower()


class BlendModel:
    """Trains Ridge first, then tree model on Ridge residuals.

    Ridge captures linear structure; tree captures nonlinear residual.
    Sequential fitting avoids model competition for the same signal.
    """

    def __init__(self):
        if _TREE_TYPE == "xgb":
            self._tree = XGBModel()
        else:
            self._tree = LGBModel()
        self._ridge = MeowModel(cacheDir=None)

    def reset(self):
        self._tree.reset()
        self._ridge.reset()

    def partial_fit(self, xdf, ydf):
        self._tree.partial_fit(xdf, ydf)
        self._ridge.partial_fit(xdf, ydf)

    def finalize_fit(self):
        # Finalize Ridge first
        self._ridge.finalize_fit()

        # Compute Ridge predictions on tree reservoir, replace targets with residuals
        if (self._tree._X_reservoir is not None
                and self._tree._feature_names is not None
                and len(self._tree._X_reservoir) > 0):
            xdf_reservoir = pd.DataFrame(
                self._tree._X_reservoir,
                columns=self._tree._feature_names,
            )
            ridge_pred = self._ridge.predict(xdf_reservoir)
            residuals = self._tree._y_reservoir - ridge_pred
            group_ids = getattr(self._tree, '_group_ids', None)
            self._tree.replace_targets(residuals, group_ids)

        # Finalize tree model on residuals
        self._tree.finalize_fit()

    def predict(self, xdf):
        ridge_pred = self._ridge.predict(xdf)
        tree_pred = self._tree.predict(xdf)
        return ridge_pred + tree_pred
