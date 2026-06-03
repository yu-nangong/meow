"""Ensemble blend: Ridge fit first, then XGB on Ridge residuals for complementary signal."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from models.xgb_model import XGBModel
from models.lgb_model import LGBModel
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

        # Compute Ridge predictions on XGB reservoir, replace targets with residuals
        if (self._tree._X_reservoir is not None
                and self._tree._feature_names is not None
                and len(self._tree._X_reservoir) > 0):
            # Build DataFrame from XGB reservoir for Ridge prediction
            xdf_reservoir = pd.DataFrame(
                self._tree._X_reservoir,
                columns=self._tree._feature_names,
            )
            ridge_pred = self._ridge.predict(xdf_reservoir)
            residuals = self._tree._y_reservoir - ridge_pred

            # Replace XGB targets with residuals
            self._tree.replace_targets(residuals, self._tree._group_ids)

        # Finalize XGB on residuals (uses its env-configured objective)
        self._tree.finalize_fit()

    def predict(self, xdf):
        ridge_pred = self._ridge.predict(xdf)
        tree_pred = self._tree.predict(xdf)
        return ridge_pred + tree_pred
