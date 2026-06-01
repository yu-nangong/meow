"""Ensemble blend: LGB + Ridge with learned stacking combiner."""
from __future__ import annotations

import os
import warnings

import numpy as np

from models.lgb_model import LGBModel
from mdl import MeowModel


class BlendModel:
    """Trains LGB and Ridge in parallel, learns optimal combination via stacking.

    During partial_fit, reservoir-samples features for stacking.
    In finalize_fit, fits a 2-feature Ridge on (LGB_pred, Ridge_pred) to
    learn optimal combination weights — replaces the fixed 0.5 blend default.
    Fallback to fixed weight if stacking is disabled or insufficient data.
    """

    def __init__(self):
        self._lgb = LGBModel()
        self._ridge = MeowModel(cacheDir=None)
        self._use_stacking = os.environ.get("MEOW_BLEND_STACKING", "1") != "0"
        self._stack_alpha = float(os.environ.get("MEOW_BLEND_STACK_ALPHA", "0.01"))
        self._stack_reservoir_size = int(os.environ.get("MEOW_BLEND_STACK_RESERVOIR", "30000"))
        self._lgb_weight = float(os.environ.get("MEOW_BLEND_LGB_WEIGHT", "0.5"))
        # Stacking reservoir state
        self._X_stack = None
        self._y_stack = None
        self._n_stack = 0
        self._stack_rng = np.random.RandomState(123)
        # Learned stacking parameters
        self._stack_w_lgb = 0.0
        self._stack_w_ridge = 0.0
        self._stack_bias = 0.0
        self._stack_fitted = False

    def reset(self):
        self._lgb.reset()
        self._ridge.reset()
        self._X_stack = None
        self._y_stack = None
        self._n_stack = 0
        self._stack_fitted = False

    def partial_fit(self, xdf, ydf):
        self._lgb.partial_fit(xdf, ydf)
        self._ridge.partial_fit(xdf, ydf)
        if self._use_stacking:
            self._reservoir_sample(xdf, ydf)

    def finalize_fit(self):
        self._lgb.finalize_fit()
        self._ridge.finalize_fit()
        if self._use_stacking and self._X_stack is not None and self._n_stack >= 1000:
            self._fit_stack()
            self._stack_fitted = True

    def predict(self, xdf):
        lgb_pred = self._lgb.predict(xdf)
        ridge_pred = self._ridge.predict(xdf)
        if self._stack_fitted:
            return (self._stack_w_lgb * lgb_pred
                    + self._stack_w_ridge * ridge_pred
                    + self._stack_bias)
        return self._lgb_weight * lgb_pred + (1.0 - self._lgb_weight) * ridge_pred

    # ---- stacking internals ----

    def _reservoir_sample(self, xdf, ydf):
        x = xdf.to_numpy(dtype=np.float32)
        y = ydf.to_numpy(dtype=np.float64).ravel()
        n = len(x)
        if self._X_stack is None:
            take = min(n, self._stack_reservoir_size)
            self._X_stack = x[:take].copy()
            self._y_stack = y[:take].copy()
            self._n_stack = take
        elif self._n_stack < self._stack_reservoir_size:
            take = min(n, self._stack_reservoir_size - self._n_stack)
            self._X_stack = np.concatenate([self._X_stack, x[:take]], axis=0)
            self._y_stack = np.concatenate([self._y_stack, y[:take]], axis=0)
            self._n_stack += take
        else:
            cap = self._stack_reservoir_size
            for i in range(n):
                j = self._stack_rng.randint(0, self._n_stack + i + 1)
                if j < cap:
                    self._X_stack[j] = x[i]
                    self._y_stack[j] = y[i]
            self._n_stack += n

    def _fit_stack(self):
        # Build DataFrame wrapper so model predict() methods work with column names
        import pandas as pd
        from feat import MeowFeatureGenerator
        # Get feature names to reconstruct DataFrame
        # Use the same order as genFeatures output
        fg = MeowFeatureGenerator(cacheDir=None)
        all_names = fg.featureNames()
        xdf_stack = pd.DataFrame(self._X_stack, columns=all_names)

        lgb_pred = self._lgb.predict(xdf_stack)
        ridge_pred = self._ridge.predict(xdf_stack)
        y = self._y_stack

        # 2-feature Ridge with bias: solve (X^T X + alpha*I) w = X^T y
        X = np.column_stack([lgb_pred, ridge_pred, np.ones(len(y))])
        XTX = X.T @ X
        XTy = X.T @ y
        penalty = self._stack_alpha * np.eye(3)
        penalty[2, 2] = 0.0  # don't regularize bias
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = np.linalg.solve(XTX + penalty, XTy)
        self._stack_w_lgb = float(w[0])
        self._stack_w_ridge = float(w[1])
        self._stack_bias = float(w[2])

        # Free reservoir memory
        self._X_stack = None
        self._y_stack = None
