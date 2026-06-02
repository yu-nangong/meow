"""Two-stage stacking: Ridge then LGB on Ridge residuals.

Ridge captures linear structure; LGB learns nonlinear patterns
that Ridge systematically misses. This is fundamentally different
from blending because LGB is trained on residuals (complementary)
rather than on the same target (competitive).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.lgb_model import LGBModel
from mdl import MeowModel


class StackingModel:
    """Train Ridge first, then LGB on Ridge residuals.

    Unlike blending (which trains both models on the same target
    and averages), stacking explicitly makes LGB a meta-model that
    learns to correct Ridge's systematic errors.
    """

    def __init__(self):
        self._lgb = LGBModel()
        self._ridge = MeowModel(cacheDir=None)

    def reset(self):
        self._lgb.reset()
        self._ridge.reset()

    def partial_fit(self, xdf, ydf):
        self._ridge.partial_fit(xdf, ydf)
        self._lgb.partial_fit(xdf, ydf)

    def finalize_fit(self):
        # 1. Finalize Ridge (trained on all data chunks)
        self._ridge.finalize_fit()

        # 2. Compute Ridge predictions on the LGB reservoir
        if self._lgb._X_reservoir is not None and self._lgb._feature_names is not None:
            x_pseudo = pd.DataFrame(
                self._lgb._X_reservoir.astype(np.float64),
                columns=self._lgb._feature_names,
            )
            ridge_reservoir_pred = self._ridge.predict(x_pseudo)
            residuals = self._lgb._y_reservoir.ravel() - ridge_reservoir_pred
            self._lgb.replace_targets(residuals)

        # 3. Train LGB on Ridge residuals
        self._lgb.finalize_fit()

    def predict(self, xdf):
        ridge_pred = self._ridge.predict(xdf)
        lgb_pred = self._lgb.predict(xdf)
        return ridge_pred + lgb_pred
