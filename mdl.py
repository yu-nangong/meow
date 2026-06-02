"""MEOW Ridge model with optional PCA feature orthogonalization.

When MEOW_PCA_N_COMPONENTS > 0, features are projected onto the top-K
principal components before Ridge fitting. PCA creates perfectly
orthogonal features, eliminating collinearity structurally.
"""
import os
import numpy as np
from log import log


class MeowModel(object):
    def __init__(self, cacheDir):
        self.alpha = float(os.environ.get("MEOW_RIDGE_ALPHA", "0.0015"))
        self.base_alpha_mult = float(os.environ.get("MEOW_BASE_ALPHA_MULT", "1.0"))
        self.cs_alpha_mult = float(os.environ.get("MEOW_CS_ALPHA_MULT", "1.0"))
        self.rank_alpha_mult = float(os.environ.get("MEOW_RANK_ALPHA_MULT", "1.0"))
        self.time_basis_alpha_mult = float(os.environ.get("MEOW_TIME_BASIS_ALPHA_MULT", "1.0"))
        self.time_alpha_mult = float(os.environ.get("MEOW_TIME_ALPHA_MULT", "1.0"))
        self.u_alpha_mult = float(os.environ.get("MEOW_U_ALPHA_MULT", "1.0"))
        self.time_sq_alpha_mult = float(os.environ.get("MEOW_TIME_SQ_ALPHA_MULT", "1.0"))
        self.u_sq_alpha_mult = float(os.environ.get("MEOW_U_SQ_ALPHA_MULT", "1.0"))
        self.exclude_families = {
            family.strip()
            for family in os.environ.get("MEOW_EXCLUDE_FAMILIES", "cs").split(",")
            if family.strip()
        }
        self.exclude_patterns = tuple(
            pattern.strip() for pattern in os.environ.get("MEOW_EXCLUDE_PATTERNS", "").split(",") if pattern.strip()
        )
        self.pca_n_components = int(os.environ.get("MEOW_PCA_N_COMPONENTS", "0"))
        self.pca_alpha = float(os.environ.get("MEOW_PCA_ALPHA", "0.003"))
        self._XtX = None
        self._Xty = None
        self._n_features = None
        self._feature_names = None
        self._selected_columns = None
        self._sum_x = None
        self._sum_x2 = None
        self._sum_y = 0.0
        self._n_rows = 0
        self._mean_x = None
        self._scale_x = None
        self._coef = None
        self._intercept = 0.0
        self._pca_V = None
        self._pca_coef = None

    def reset(self):
        self._XtX = None
        self._Xty = None
        self._n_features = None
        self._feature_names = None
        self._selected_columns = None
        self._sum_x = None
        self._sum_x2 = None
        self._sum_y = 0.0
        self._n_rows = 0
        self._mean_x = None
        self._scale_x = None
        self._coef = None
        self._intercept = 0.0
        self._pca_V = None
        self._pca_coef = None

    def partial_fit(self, xdf, ydf):
        xdf = self._select_columns(xdf)
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

        if self.pca_n_components > 0 and self._n_features > self.pca_n_components:
            self._fit_pca(centered_xtx, centered_xty, mean_x, inv_scale)
        else:
            self._fit_direct(centered_xtx, centered_xty, mean_x, inv_scale)

        self._mean_x = mean_x
        self._scale_x = scale_x
        log.inf("Done fitting")

    def _fit_direct(self, centered_xtx, centered_xty, mean_x, inv_scale):
        ztz = centered_xtx * np.outer(inv_scale, inv_scale)
        zty = centered_xty * inv_scale
        ridge_diag = self._ridge_diag()
        coef_scaled = np.linalg.solve(ztz + np.diag(ridge_diag), zty)
        self._coef = coef_scaled * inv_scale
        mean_y = self._sum_y / max(self._n_rows, 1)
        self._intercept = float(mean_y - mean_x @ self._coef)

    def _fit_pca(self, centered_xtx, centered_xty, mean_x, inv_scale):
        n = self._n_features
        k = min(self.pca_n_components, n)
        inv_2d = np.outer(inv_scale, inv_scale)
        ztz = centered_xtx * inv_2d
        zty = centered_xty * inv_scale
        evals, evecs = np.linalg.eigh(ztz)
        V = evecs[:, -k:]  # top k eigenvectors (largest eigenvalues last)
        zty_pca = V.T @ zty
        evals_K = np.diag(V.T @ ztz @ V)
        denom = evals_K + self.pca_alpha
        coef_pca = zty_pca / np.maximum(denom, 1e-12)
        self._pca_V = V
        self._pca_coef = coef_pca
        orig_coef = V @ coef_pca * inv_scale
        mean_y = self._sum_y / max(self._n_rows, 1)
        self._intercept = float(mean_y - mean_x @ orig_coef)

    def _ridge_diag(self):
        ridge_diag = np.full(self._n_features, self.alpha, dtype=np.float64)
        for idx, name in enumerate(self._feature_names):
            if name.endswith("_x_time_sq"):
                ridge_diag[idx] *= self.time_sq_alpha_mult
            elif name.endswith("_x_u_sq"):
                ridge_diag[idx] *= self.u_sq_alpha_mult
            elif name.endswith("_x_time"):
                ridge_diag[idx] *= self.time_alpha_mult
            elif name.endswith("_x_u"):
                ridge_diag[idx] *= self.u_alpha_mult
            elif name.endswith("_rank_cs"):
                ridge_diag[idx] *= self.rank_alpha_mult
            elif name.endswith("_cs"):
                ridge_diag[idx] *= self.cs_alpha_mult
            elif name.startswith("interval_"):
                ridge_diag[idx] *= self.time_basis_alpha_mult
            else:
                ridge_diag[idx] *= self.base_alpha_mult
        return ridge_diag

    def fit(self, xdf, ydf):
        self.reset()
        self.partial_fit(xdf, ydf)
        self.finalize_fit()

    def predict(self, xdf):
        if self._pca_V is not None:
            return self._predict_pca(xdf)
        return self._predict_direct(xdf)

    def _predict_direct(self, xdf):
        xdf = self._select_columns(xdf)
        x = xdf.to_numpy(dtype=np.float64)
        return x @ self._coef + self._intercept

    def _predict_pca(self, xdf):
        xdf = self._select_columns(xdf)
        x = xdf.to_numpy(dtype=np.float64)
        z = (x - self._mean_x) * (1.0 / self._scale_x)
        z_pca = z @ self._pca_V
        return z_pca @ self._pca_coef + self._intercept

    def _select_columns(self, xdf):
        if not self.exclude_patterns:
            if not self.exclude_families:
                return xdf
        if not self.exclude_patterns and not self.exclude_families:
            return xdf
        if self._selected_columns is None:
            self._selected_columns = [
                col
                for col in xdf.columns
                if self._keep_column(col)
            ]
        return xdf.loc[:, self._selected_columns]

    def _keep_column(self, name):
        if self.exclude_patterns and any(pattern in name for pattern in self.exclude_patterns):
            return False
        return self._family_of(name) not in self.exclude_families

    @staticmethod
    def _family_of(name):
        if name.endswith("_x_time_sq"):
            return "time_sq_interaction"
        if name.endswith("_x_u_sq"):
            return "u_sq_interaction"
        if name.endswith("_x_time"):
            return "time_interaction"
        if name.endswith("_x_u"):
            return "u_interaction"
        if name.endswith("_rank_cs"):
            return "rank"
        if name.endswith("_cs"):
            return "cs"
        if name.startswith("interval_"):
            return "time_basis"
        return "raw"
