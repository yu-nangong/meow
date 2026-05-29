import os
from types import SimpleNamespace

import numpy as np
from log import log


class MeowModel(object):
    def __init__(self, cacheDir):
        self.alpha = float(os.environ.get("MEOW_RIDGE_ALPHA", "0.001"))
        self.interval_experts = max(int(os.environ.get("MEOW_INTERVAL_EXPERTS", "3")), 1)
        self.interval_expert_blend = float(os.environ.get("MEOW_INTERVAL_EXPERT_BLEND", "0.5"))
        self.base_alpha_mult = float(os.environ.get("MEOW_BASE_ALPHA_MULT", "1.0"))
        self.cs_alpha_mult = float(os.environ.get("MEOW_CS_ALPHA_MULT", "1.0"))
        self.rank_alpha_mult = float(os.environ.get("MEOW_RANK_ALPHA_MULT", "1.0"))
        self.time_basis_alpha_mult = float(os.environ.get("MEOW_TIME_BASIS_ALPHA_MULT", "1.0"))
        self.time_alpha_mult = float(os.environ.get("MEOW_TIME_ALPHA_MULT", "0.35"))
        self.u_alpha_mult = float(os.environ.get("MEOW_U_ALPHA_MULT", "0.35"))
        self.time_sq_alpha_mult = float(os.environ.get("MEOW_TIME_SQ_ALPHA_MULT", "0.6"))
        self.u_sq_alpha_mult = float(os.environ.get("MEOW_U_SQ_ALPHA_MULT", "0.6"))
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
        self._expert_states = []
        self._expert_coef = []
        self._expert_intercept = []

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
        self._expert_states = []
        self._expert_coef = []
        self._expert_intercept = []

    def partial_fit(self, xdf, ydf):
        expert_codes = self._expert_bucket_codes(xdf)
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
            self._expert_states = [self._make_state(self._n_features) for _ in range(self.interval_experts)]
        self._update_state(self, x, y)
        if self.interval_experts > 1:
            for bucket, state in enumerate(self._expert_states):
                mask = expert_codes == bucket
                if np.any(mask):
                    self._update_state(state, x[mask], y[mask])

    def finalize_fit(self):
        self._coef, self._intercept, self._mean_x, self._scale_x = self._solve_state(self)
        self._expert_coef = []
        self._expert_intercept = []
        for state in self._expert_states:
            coef, intercept, _, _ = self._solve_state(state, fallback_coef=self._coef, fallback_intercept=self._intercept)
            self._expert_coef.append(coef)
            self._expert_intercept.append(intercept)
        log.inf("Done fitting")

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
        expert_codes = self._expert_bucket_codes(xdf)
        xdf = self._select_columns(xdf)
        x = xdf.to_numpy(dtype=np.float64)
        pred = x @ self._coef + self._intercept
        if self.interval_experts <= 1:
            return pred
        expert_pred = pred.copy()
        for bucket, (coef, intercept) in enumerate(zip(self._expert_coef, self._expert_intercept)):
            mask = expert_codes == bucket
            if np.any(mask):
                expert_pred[mask] = x[mask] @ coef + intercept
        blend = np.clip(self.interval_expert_blend, 0.0, 1.0)
        return (1.0 - blend) * pred + blend * expert_pred

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

    def _expert_bucket_codes(self, xdf):
        if self.interval_experts <= 1 or "interval_frac_centered" not in xdf.columns:
            return np.zeros(len(xdf), dtype=np.int64)
        frac = np.clip(xdf["interval_frac_centered"].to_numpy(dtype=np.float64, copy=False) + 0.5, 0.0, 1.0 - 1e-12)
        return np.minimum((frac * self.interval_experts).astype(np.int64), self.interval_experts - 1)

    @staticmethod
    def _make_state(n_features):
        return SimpleNamespace(
            _XtX=np.zeros((n_features, n_features), dtype=np.float64),
            _Xty=np.zeros(n_features, dtype=np.float64),
            _sum_x=np.zeros(n_features, dtype=np.float64),
            _sum_x2=np.zeros(n_features, dtype=np.float64),
            _sum_y=0.0,
            _n_rows=0,
        )

    @staticmethod
    def _update_state(state, x, y):
        state._XtX += x.T @ x
        state._Xty += x.T @ y
        state._sum_x += x.sum(axis=0)
        state._sum_x2 += np.square(x).sum(axis=0)
        state._sum_y += y.sum()
        state._n_rows += len(y)

    def _solve_state(self, state, fallback_coef=None, fallback_intercept=None):
        if state._n_rows <= max(self._n_features, 1):
            if fallback_coef is not None and fallback_intercept is not None:
                return fallback_coef.copy(), float(fallback_intercept), None, None
        mean_x = state._sum_x / max(state._n_rows, 1)
        var_x = state._sum_x2 / max(state._n_rows, 1) - np.square(mean_x)
        scale_x = np.sqrt(np.maximum(var_x, 1e-12))
        inv_scale = 1.0 / scale_x
        centered_xtx = state._XtX - state._n_rows * np.outer(mean_x, mean_x)
        centered_xty = state._Xty - mean_x * state._sum_y
        ztz = centered_xtx * np.outer(inv_scale, inv_scale)
        zty = centered_xty * inv_scale
        ridge_diag = self._ridge_diag()
        coef_scaled = np.linalg.solve(ztz + np.diag(ridge_diag), zty)
        mean_y = state._sum_y / max(state._n_rows, 1)
        coef = coef_scaled * inv_scale
        intercept = float(mean_y - mean_x @ coef)
        return coef, intercept, mean_x, scale_x

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
