"""Raw LOB LightGBM residual: trains LGB on raw order book columns
(normalized price levels, log-transformed sizes/turnovers/counts)
to capture nonlinear book-shape patterns the linear Ridge pipeline misses.

Fundamentally different from all prior attempts:
- Uses raw LOB data, NOT the ~250 engineered features
- Tree model discovers nonlinear interactions between bid/ask book levels
- Operates as a residual on top of the existing Ridge pipeline
"""
from __future__ import annotations

import os

import numpy as np
import lightgbm as lgb

# Price-like columns: normalize by midpx for cross-stock comparability
_PRICE_COLS = [
    "bid0", "ask0", "bid4", "ask4", "bid9", "ask9", "bid19", "ask19",
    "midpx", "lastpx", "open", "high", "low",
    "buyVwad", "sellVwad",
    "tradeBuyHigh", "tradeBuyLow", "tradeSellHigh", "tradeSellLow",
    "addBuyHigh", "addBuyLow", "addSellHigh", "addSellLow",
    "cxlBuyHigh", "cxlBuyLow", "cxlSellHigh", "cxlSellLow",
]

# Quantity-like columns: log1p transform for scale invariance
_QUANTITY_COLS = [
    "bsize0", "asize0",
    "bsize0_4", "asize0_4", "bsize5_9", "asize5_9", "bsize10_19", "asize10_19",
    "tradeBuyQty", "tradeSellQty", "tradeBuyTurnover", "tradeSellTurnover",
    "addBuyQty", "addSellQty", "addBuyTurnover", "addSellTurnover",
    "cxlBuyQty", "cxlSellQty", "cxlBuyTurnover", "cxlSellTurnover",
]

# Count columns: log1p transform
_COUNT_COLS = [
    "nTradeBuy", "nTradeSell", "nAddBuy", "nAddSell", "nCxlBuy", "nCxlSell",
]

# Turnover depth columns: log1p transform
_TURNOVER_COLS = [
    "btr0_4", "atr0_4", "btr5_9", "atr5_9", "btr10_19", "atr10_19",
]


class RawLOBResidual:
    """LightGBM residual model trained on raw LOB columns.

    Takes the raw DataFrame (not engineered features), normalizes columns,
    and fits LGB on the residuals of the existing linear pipeline.
    """

    def __init__(self):
        self.max_rows = int(os.environ.get("MEOW_RAWLOB_MAX_ROWS", "400000"))
        self.num_leaves = int(os.environ.get("MEOW_RAWLOB_NUM_LEAVES", "15"))
        self.learning_rate = float(os.environ.get("MEOW_RAWLOB_LR", "0.03"))
        self.n_estimators = int(os.environ.get("MEOW_RAWLOB_N_ESTIMATORS", "150"))
        self.subsample = float(os.environ.get("MEOW_RAWLOB_SUBSAMPLE", "0.7"))
        self.min_child_samples = int(os.environ.get("MEOW_RAWLOB_MIN_CHILD", "200"))

        self._X_reservoir = None
        self._y_reservoir = None
        self._n_accumulated = 0
        self._model = None
        self._active_cols = None
        self._price_idx = None
        self._mean = None
        self._std = None
        self._rng = np.random.RandomState(42)

    def reset(self):
        self._X_reservoir = None
        self._y_reservoir = None
        self._n_accumulated = 0
        self._model = None
        self._active_cols = None
        self._price_idx = None
        self._mean = None
        self._std = None

    def partial_fit(self, raw, residual):
        """Accumulate normalized raw LOB data and target residuals.

        Args:
            raw: pandas DataFrame with raw HDF5 columns
            residual: 1D numpy array of forecast residuals to predict
        """
        features = self._extract_features(raw)
        if features is None or len(features) == 0:
            return
        n = len(features)
        if self._X_reservoir is None:
            take = min(n, self.max_rows)
            self._X_reservoir = features[:take].copy()
            self._y_reservoir = residual[:take].astype(np.float32, copy=True)
            self._n_accumulated = n
            return
        capacity = self._X_reservoir.shape[0]
        if capacity < self.max_rows:
            take = min(n, self.max_rows - capacity)
            self._X_reservoir = np.concatenate(
                [self._X_reservoir, features[:take]], axis=0
            )
            self._y_reservoir = np.concatenate(
                [self._y_reservoir, residual[:take].astype(np.float32, copy=True)], axis=0
            )
            start = take
        else:
            start = 0
        for i in range(start, n):
            j = self._rng.randint(0, self._n_accumulated + i + 1)
            if j < capacity:
                self._X_reservoir[j] = features[i]
                self._y_reservoir[j] = residual[i]
        self._n_accumulated += n

    def finalize_fit(self):
        if self._X_reservoir is None or len(self._y_reservoir) < 5000:
            return
        self._mean = self._X_reservoir.mean(axis=0, dtype=np.float64)
        self._std = self._X_reservoir.std(axis=0, dtype=np.float64)
        self._std = np.where(self._std > 1e-8, self._std, 1.0)
        x_train = (
            (self._X_reservoir - self._mean) / self._std
        ).astype(np.float32, copy=False)

        params = dict(
            boosting_type="gbdt",
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            subsample_freq=1,
            min_child_samples=self.min_child_samples,
            reg_lambda=1.0,
            verbose=-1,
            random_state=42,
            n_jobs=1,
        )
        train_data = lgb.Dataset(x_train, label=self._y_reservoir, free_raw_data=False)
        self._model = lgb.train(params, train_data, num_boost_round=self.n_estimators)

    def predict(self, raw):
        if self._model is None:
            return np.zeros(len(raw), dtype=np.float64)
        features = self._extract_features(raw)
        if features is None:
            return np.zeros(len(raw), dtype=np.float64)
        features = (
            (features - self._mean) / self._std
        ).astype(np.float32, copy=False)
        return self._model.predict(features).astype(np.float64)

    def _extract_features(self, raw):
        """Normalize raw LOB columns into a feature matrix."""
        if self._active_cols is None:
            self._active_cols = []
            self._price_idx = []
            for col in (
                _PRICE_COLS + _QUANTITY_COLS + _COUNT_COLS + _TURNOVER_COLS
            ):
                if col in raw.columns:
                    self._active_cols.append(col)
            self._price_idx = [
                i for i, col in enumerate(self._active_cols) if col in _PRICE_COLS
            ]
            self._midpx_idx = (
                self._active_cols.index("midpx") if "midpx" in self._active_cols else None
            )

        if not self._active_cols:
            return None

        values = raw.loc[:, self._active_cols].to_numpy(dtype=np.float64, copy=True)
        values = np.where(np.isfinite(values), values, 0.0)

        # Normalize price columns by midpx
        if self._midpx_idx is not None and self._price_idx:
            mid = np.maximum(np.abs(values[:, self._midpx_idx : self._midpx_idx + 1]), 1e-6)
            values[:, self._price_idx] = values[:, self._price_idx] / mid

        # Log-transform quantity/count/turnover columns
        non_price = np.ones(values.shape[1], dtype=bool)
        if self._price_idx:
            non_price[self._price_idx] = False
        if self._midpx_idx is not None:
            non_price[self._midpx_idx] = False  # midpx already served as denominator

        values[:, non_price] = np.sign(values[:, non_price]) * np.log1p(
            np.abs(values[:, non_price])
        )

        # Replace any remaining inf/nan
        values = np.where(np.isfinite(values), values, 0.0)
        return values.astype(np.float32, copy=False)
