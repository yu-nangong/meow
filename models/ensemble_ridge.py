"""Random subspace Ridge ensemble.

Trains multiple Ridge models on random feature subsets and averages
predictions. Reduces variance from feature collinearity by diversifying
across feature subspaces — a fundamentally different model structure
from the single Ridge+LGB blend.
"""
from __future__ import annotations

import os

import numpy as np

from mdl import MeowModel


class EnsembleRidgeModel:
    """N Ridge models trained on random feature subspaces, averaged."""

    def __init__(self):
        self._n_models = int(os.environ.get("MEOW_RIDGE_ENSEMBLE", "1"))
        self._subset_frac = float(os.environ.get("MEOW_RIDGE_ENSEMBLE_FRAC", "0.6"))
        self._rng = np.random.RandomState(42)
        self._models = [MeowModel(cacheDir=None) for _ in range(self._n_models)]
        self._feature_subsets = None

    def reset(self):
        self._feature_subsets = None
        for model in self._models:
            model.reset()

    def partial_fit(self, xdf, ydf):
        if self._feature_subsets is None and self._n_models > 1:
            all_cols = list(xdf.columns)
            n_total = len(all_cols)
            n_subset = max(1, int(n_total * self._subset_frac))
            self._feature_subsets = []
            for _ in range(self._n_models):
                idx = self._rng.choice(n_total, n_subset, replace=False)
                self._feature_subsets.append([all_cols[i] for i in sorted(idx)])
        if self._n_models == 1:
            self._models[0].partial_fit(xdf, ydf)
        else:
            for i, model in enumerate(self._models):
                sub_xdf = xdf[self._feature_subsets[i]]
                model.partial_fit(sub_xdf, ydf)

    def finalize_fit(self):
        for model in self._models:
            model.finalize_fit()

    def predict(self, xdf):
        if self._n_models == 1:
            return self._models[0].predict(xdf)
        preds = np.column_stack(
            [
                model.predict(xdf[self._feature_subsets[i]])
                for i, model in enumerate(self._models)
            ]
        )
        return preds.mean(axis=1, dtype=np.float64)
