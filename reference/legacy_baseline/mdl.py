import os
import numpy as np
from sklearn.linear_model import Ridge
from log import log


class MeowModel(object):
    def __init__(self, cacheDir):
        self.estimator = Ridge(
            alpha=0.5,
            random_state=None,
            fit_intercept=False,
            tol=1e-8
        )
        self._XtX = None
        self._Xty = None
        self._n_features = None

    def reset(self):
        self._XtX = None
        self._Xty = None
        self._n_features = None

    def partial_fit(self, xdf, ydf):
        x = xdf.to_numpy(dtype=np.float64)
        y = ydf.to_numpy(dtype=np.float64).ravel()
        if self._XtX is None:
            self._n_features = x.shape[1]
            self._XtX = np.zeros((self._n_features, self._n_features), dtype=np.float64)
            self._Xty = np.zeros(self._n_features, dtype=np.float64)
        self._XtX += x.T @ x
        self._Xty += x.T @ y

    def finalize_fit(self):
        alpha = self.estimator.alpha
        coef = np.linalg.solve(
            self._XtX + alpha * np.eye(self._n_features),
            self._Xty,
        )
        self.estimator.coef_ = coef
        self.estimator.intercept_ = np.zeros(1, dtype=np.float64)
        self.estimator.n_features_in_ = self._n_features
        log.inf("Done fitting")

    def fit(self, xdf, ydf):
        self.reset()
        self.partial_fit(xdf, ydf)
        self.finalize_fit()

    def predict(self, xdf):
        return self.estimator.predict(xdf.to_numpy())
