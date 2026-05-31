"""
Random Kitchen Sink (RKS) features for MEOW.

Random Fourier Features approximate an RBF kernel:
    k(x, y) ≈ (1/D) * sum_{j=1}^{D} cos(w_j^T x + b_j)

where w_j ~ N(0, 1/sigma^2) and b_j ~ Uniform(0, 2*pi).

This creates nonlinear features that can be fed into any linear model
(ridge, elasticnet), adding nonlinear capacity without a nonlinear model.
"""

from __future__ import annotations

import os
import numpy as np


class RKSFeatureGenerator:
    """Generates random Fourier features to approximate an RBF kernel."""

    def __init__(self, n_features: int = 30, gamma: float | None = None, seed: int = 42):
        self.n_features = int(os.environ.get("MEOW_RKS_N_FEATURES", str(n_features)))
        gamma_env = os.environ.get("MEOW_RKS_GAMMA", "")
        self.gamma = float(gamma_env) if gamma_env else gamma
        self.seed = int(os.environ.get("MEOW_RKS_SEED", str(seed)))
        self._rng = np.random.RandomState(self.seed)
        self._omega: np.ndarray | None = None  # random projection matrix
        self._b: np.ndarray | None = None  # random biases
        self._input_dim: int | None = None
        self._fitted = False

    def fit(self, x: np.ndarray):
        """Learn random projection matrix from training data.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_input_features)
            Training data used to determine input dimensionality and gamma.
        """
        n_samples, d = x.shape
        self._input_dim = d

        # Determine gamma: if not provided, use a heuristic
        if self.gamma is None:
            # Heuristic: 1 / (median pairwise distance) or 1/d
            self.gamma = 1.0 / float(d)

        # Sample random projection matrix: w_j ~ N(0, 2*gamma)
        # For RBF kernel k(x,y) = exp(-gamma * ||x-y||^2), the Fourier dual
        # is N(0, 2*gamma). See Rahimi & Recht 2007.
        std = np.sqrt(2.0 * self.gamma)
        self._omega = self._rng.normal(
            loc=0.0,
            scale=std,
            size=(d, self.n_features),
        ).astype(np.float64)

        # Uniform biases b_j ~ Uniform(0, 2*pi)
        self._b = self._rng.uniform(
            low=0.0,
            high=2.0 * np.pi,
            size=self.n_features,
        ).astype(np.float64)

        self._fitted = True

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Project x through random Fourier features.

        Returns cos(X @ omega + b) with shape (n_samples, n_features).
        """
        if not self._fitted or self._omega is None or self._b is None:
            raise ValueError("RKSFeatureGenerator not fitted")
        return np.cos(x @ self._omega + self._b[None, :]).astype(np.float64)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)

    @property
    def output_dim(self) -> int:
        return self.n_features


def make_rks_feature_names(rks: RKSFeatureGenerator, prefix: str = "rks") -> list[str]:
    """Generate human-readable names for RKS features."""
    return [f"{prefix}_{i}" for i in range(rks.n_features)]
