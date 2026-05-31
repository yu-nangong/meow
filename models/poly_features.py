"""
Fixed-top-K polynomial feature expansion for MEOW.

Adds squared + pairwise interaction terms for the top features
identified in a probe run. Controlled by env vars:

MEOW_POLY_ENABLE=1          # Enable polynomial features
MEOW_POLY_SQUARES=1         # Add x^2 for each top feature
MEOW_POLY_INTERACTIONS=1    # Add x_i * x_j for pairs of top features
MEOW_POLY_TOP_K=10          # How many top features to expand

The top features are identified by their names. If a feature name isn't
present in the dataframe, it's silently skipped.
"""
from __future__ import annotations

import os
from itertools import combinations

import numpy as np
import pandas as pd


# Top features identified from probe run on 8 days of training data
# These are the 15 features with highest |coefficient| from ElasticNet
_DEFAULT_TOP_FEATURES = [
    "high_minus_low",
    "micro_dev_cs",
    "micro_dev_ema6",
    "ret1_cs",
    "ret24",
    "ret6_cs",
    "ret3_cs",
    "ret12_resid",
    "spread_cs",
    "interval_frac_centered",
    "interval_u",
    "ret3_rank_cs_x_u",
    "high_minus_low_rank_cs_x_time",
    "ret12_cs",
    "trade_imb_ema6",
    "ret12_resid_cs",
    "add_count_imb",
    "trade_count_share_rank_cs",
    "flow_imb",
    "ret1_rank_cs",
]


class PolyFeatureExpander:
    def __init__(self):
        raw_list = os.environ.get(
            "MEOW_POLY_FEATURES",
            ",".join(_DEFAULT_TOP_FEATURES),
        )
        self._target_features = [name.strip() for name in raw_list.split(",") if name.strip()]
        self.top_k = int(os.environ.get("MEOW_POLY_TOP_K", "10"))
        self.add_squares = os.environ.get("MEOW_POLY_SQUARES", "1") != "0"
        self.add_interactions = os.environ.get("MEOW_POLY_INTERACTIONS", "1") != "0"
        self._fitted = False

    @property
    def is_enabled(self) -> bool:
        return os.environ.get("MEOW_POLY_ENABLE", "0") != "0"

    def transform(self, xdf: pd.DataFrame) -> pd.DataFrame:
        """Add polynomial features to the dataframe."""
        if not self.is_enabled or not self._target_features:
            return xdf

        available = [name for name in self._target_features[:self.top_k] if name in xdf.columns]
        if not available:
            return xdf

        k = len(available)
        parts: list[pd.DataFrame] = []

        if self.add_squares:
            sq_data = {}
            for name in available:
                arr = xdf[name].to_numpy(dtype=np.float64, copy=False)
                sq_data[f"{name}_sq"] = arr * arr
            parts.append(pd.DataFrame(sq_data, index=xdf.index))

        if self.add_interactions and k >= 2:
            int_data = {}
            count = 0
            max_int = int(os.environ.get("MEOW_POLY_MAX_INTERACTIONS", "200"))
            for i, j in combinations(range(k), 2):
                if count >= max_int:
                    break
                name_i, name_j = available[i], available[j]
                xi = xdf[name_i].to_numpy(dtype=np.float64, copy=False)
                xj = xdf[name_j].to_numpy(dtype=np.float64, copy=False)
                int_data[f"{name_i}_x_{name_j}"] = xi * xj
                count += 1
            if int_data:
                parts.append(pd.DataFrame(int_data, index=xdf.index))

        if not parts:
            return xdf

        return pd.concat([xdf] + parts, axis=1)
