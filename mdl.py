import os
import numpy as np
from log import log


class MeowModel(object):
    def __init__(self, cacheDir):
        self.alpha = float(os.environ.get("MEOW_RIDGE_ALPHA", "0.25"))
        self.time_alpha_mult = float(os.environ.get("MEOW_TIME_ALPHA_MULT", "0.35"))
        self.u_alpha_mult = float(os.environ.get("MEOW_U_ALPHA_MULT", "0.35"))
        self._XtX = None
        self._Xty = None
        self._n_features = None
        self._feature_names = None
        self._sum_x = None
        self._sum_x2 = None
        self._sum_y = 0.0
        self._n_rows = 0
        self._mean_x = None
        self._scale_x = None
        self._coef = None
        self._intercept = 0.0

    def reset(self):
        self._XtX = None
        self._Xty = None
        self._n_features = None
        self._feature_names = None
        self._sum_x = None
        self._sum_x2 = None
        self._sum_y = 0.0
        self._n_rows = 0
        self._mean_x = None
        self._scale_x = None
        self._coef = None
        self._intercept = 0.0

    def partial_fit(self, xdf, ydf):
        x = xdf.to_numpy(dtype=np.float64)
        y = ydf.to_numpy(dtype=np.float64).ravel()
        if self._XtX is None:
            self._n_features = x.shape[1]
            self._feature_names = list(xdf.columns)
            self._XtX = np.zeros((self._n_features, self._n_features), dtype=np.float64)
            self._Xty = np.zeros(self._n_features, dtype=np.float64)
            self._sum_x = np.zeros(self._n_features, dtype=np.float64)
            self._sum_x2 = np.zeros(self._n_features, dtype=np.float64)
        self._XtX += x.T @ x
        self._Xty += x.T @ y
        self._sum_x += x.sum(axis=0)
        self._sum_x2 += np.square(x).sum(axis=0)
        self._sum_y += y.sum()
        self._n_rows += len(y)

    def finalize_fit(self):
        mean_x = self._sum_x / max(self._n_rows, 1)
        var_x = self._sum_x2 / max(self._n_rows, 1) - np.square(mean_x)
        scale_x = np.sqrt(np.maximum(var_x, 1e-12))
        inv_scale = 1.0 / scale_x

        centered_xtx = self._XtX - self._n_rows * np.outer(mean_x, mean_x)
        centered_xty = self._Xty - mean_x * self._sum_y
        ztz = centered_xtx * np.outer(inv_scale, inv_scale)
        zty = centered_xty * inv_scale
        ridge_diag = self._ridge_diag()

        coef_scaled = np.linalg.solve(
            ztz + np.diag(ridge_diag),
            zty,
        )
        mean_y = self._sum_y / max(self._n_rows, 1)
        coef = coef_scaled * inv_scale

        self._mean_x = mean_x
        self._scale_x = scale_x
        self._coef = coef
        self._intercept = float(mean_y - mean_x @ coef)
        log.inf("Done fitting")

    def _ridge_diag(self):
        ridge_diag = np.full(self._n_features, self.alpha, dtype=np.float64)
        for idx, name in enumerate(self._feature_names):
            if name.endswith("_x_time"):
                ridge_diag[idx] *= self.time_alpha_mult
            elif name.endswith("_x_u"):
                ridge_diag[idx] *= self.u_alpha_mult
        return ridge_diag

    def fit(self, xdf, ydf):
        self.reset()
        self.partial_fit(xdf, ydf)
        self.finalize_fit()

    def predict(self, xdf):
        x = xdf.to_numpy(dtype=np.float64)
        return x @ self._coef + self._intercept
