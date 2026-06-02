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
        self.l1_ratio = float(os.environ.get("MEOW_ELASTIC_L1_RATIO", "0.0"))
        self.cd_max_iter = int(os.environ.get("MEOW_ELASTIC_CD_MAX_ITER", "1000"))
        self.cd_tol = float(os.environ.get("MEOW_ELASTIC_CD_TOL", "1e-4"))
        self.exclude_families = {
            family.strip()
            for family in os.environ.get("MEOW_EXCLUDE_FAMILIES", "cs").split(",")
            if family.strip()
        }
        self.exclude_patterns = tuple(
            pattern.strip() for pattern in os.environ.get("MEOW_EXCLUDE_PATTERNS", "").split(",") if pattern.strip()
        )
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
        ztz = centered_xtx * np.outer(inv_scale, inv_scale)
        zty = centered_xty * inv_scale
        ridge_diag = self._ridge_diag()

        if self.l1_ratio > 0:
            coef_scaled = self._cd_elasticnet(ztz, zty, ridge_diag)
        else:
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


    def _cd_elasticnet(self, ztz, zty, ridge_diag):
        """Coordinate descent for ElasticNet on accumulated (ztz, zty).

        Objective: min_beta ||zty - ztz @ beta||^2
          + sum_j [l1 * |beta_j| + l2_j * beta_j^2]
        where l1 = alpha * l1_ratio, l2_j = alpha * (1-l1_ratio) * mult_j.
        """
        n_feat = len(zty)
        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1.0 - self.l1_ratio) * np.asarray(ridge_diag, dtype=np.float64)
        denom = np.diag(ztz).copy() + l2

        # Warm-start from Ridge solution
        ridge_mat = ztz + np.diag(ridge_diag)
        beta = np.linalg.solve(ridge_mat, zty)

        # Residual: r = zty - ztz @ beta
        r = zty - ztz @ beta

        for iteration in range(self.cd_max_iter):
            max_delta = 0.0
            for j in range(n_feat):
                beta_old = beta[j]
                r_j = r[j] + ztz[j, j] * beta_old
                beta_new = np.sign(r_j) * max(abs(r_j) - l1, 0.0) / denom[j]
                delta = beta_new - beta_old
                beta[j] = beta_new
                if delta != 0.0:
                    r -= ztz[:, j] * delta
                max_delta = max(max_delta, abs(delta))
            if max_delta < self.cd_tol:
                break
        return beta

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
        xdf = self._select_columns(xdf)
        x = xdf.to_numpy(dtype=np.float64)
        return x @ self._coef + self._intercept

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
