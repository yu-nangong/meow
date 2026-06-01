"""Small residual tree on normalized raw snapshot columns."""
from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd


class RawResidualModel:
    def __init__(self):
        self.enabled = os.environ.get("MEOW_RAW_RESIDUAL_ENABLE", "1") != "0"
        self.max_weight = float(os.environ.get("MEOW_RAW_RESIDUAL_WEIGHT", "0.18"))
        self.calib_frac = float(os.environ.get("MEOW_RAW_RESIDUAL_CALIB_FRAC", "0.2"))
        self.min_calib_rows = int(os.environ.get("MEOW_RAW_RESIDUAL_MIN_CALIB_ROWS", "5000"))
        self.max_rows = int(os.environ.get("MEOW_RAW_RESIDUAL_MAX_ROWS", "250000"))
        self.num_leaves = int(os.environ.get("MEOW_RAW_RESIDUAL_NUM_LEAVES", "15"))
        self.learning_rate = float(os.environ.get("MEOW_RAW_RESIDUAL_LR", "0.05"))
        self.n_estimators = int(os.environ.get("MEOW_RAW_RESIDUAL_N_ESTIMATORS", "120"))
        self.subsample = float(os.environ.get("MEOW_RAW_RESIDUAL_SUBSAMPLE", "0.8"))
        self.colsample_bytree = float(os.environ.get("MEOW_RAW_RESIDUAL_COLSAMPLE", "0.7"))
        self.min_child_samples = int(os.environ.get("MEOW_RAW_RESIDUAL_MIN_CHILD", "200"))
        self.reg_lambda = float(os.environ.get("MEOW_RAW_RESIDUAL_REG_LAMBDA", "2.0"))
        self.raw_cols = [
            "bid0",
            "ask0",
            "bid4",
            "ask4",
            "bid9",
            "ask9",
            "bid19",
            "ask19",
            "bsize0",
            "asize0",
            "bsize0_4",
            "asize0_4",
            "bsize5_9",
            "asize5_9",
            "bsize10_19",
            "asize10_19",
            "btr0_4",
            "atr0_4",
            "btr5_9",
            "atr5_9",
            "btr10_19",
            "atr10_19",
            "midpx",
            "lastpx",
            "open",
            "high",
            "low",
            "tradeBuyQty",
            "tradeSellQty",
            "tradeBuyTurnover",
            "tradeSellTurnover",
            "buyVwad",
            "sellVwad",
            "addBuyQty",
            "addSellQty",
            "cxlBuyQty",
            "cxlSellQty",
            "nTradeBuy",
            "nTradeSell",
            "nAddBuy",
            "nAddSell",
            "nCxlBuy",
            "nCxlSell",
        ]
        self._X_reservoir = None
        self._y_reservoir = None
        self._n_seen = 0
        self._model = None
        self._feature_names = None
        self._weight = 0.0
        self._rng = np.random.RandomState(42)

    def reset(self):
        self._X_reservoir = None
        self._y_reservoir = None
        self._n_seen = 0
        self._model = None
        self._feature_names = None
        self._weight = 0.0

    def _transform_raw(self, raw: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.raw_cols if c in raw.columns]
        grp_keys = [raw["date"], raw["interval"]]
        base = raw.loc[:, cols].astype(np.float32, copy=False)
        grp = base.groupby(grp_keys, sort=False)
        means = grp.transform("mean")
        stds = grp.transform("std").fillna(0.0).clip(lower=1e-6)
        z = (base - means) / stds
        z.columns = [f"{c}_raw_zs" for c in cols]
        ranks = base.groupby(grp_keys, sort=False).rank(pct=True) - 0.5
        ranks.columns = [f"{c}_raw_rank" for c in cols]
        interval_max = raw.groupby("date", sort=False)["interval"].transform("max").clip(lower=1)
        frac = (
            raw["interval"].to_numpy(dtype=np.float32, copy=False)
            / interval_max.to_numpy(dtype=np.float32, copy=False)
            - 0.5
        )
        out = pd.concat([z, ranks], axis=1)
        out["interval_frac_centered"] = frac
        out["interval_abs_centered"] = np.abs(frac)
        return out.fillna(0.0)

    def partial_fit(self, raw: pd.DataFrame, resid: np.ndarray):
        if not self.enabled or self.max_weight == 0.0:
            return
        feats = self._transform_raw(raw)
        x = feats.to_numpy(dtype=np.float32, copy=False)
        y = np.asarray(resid, dtype=np.float32)
        if self._feature_names is None:
            self._feature_names = list(feats.columns)
        n = len(x)
        if self._X_reservoir is None:
            take = min(n, self.max_rows)
            self._X_reservoir = x[:take].copy()
            self._y_reservoir = y[:take].copy()
            self._n_seen = n
            return
        capacity = self._X_reservoir.shape[0]
        if capacity < self.max_rows:
            take = min(n, self.max_rows - capacity)
            self._X_reservoir = np.concatenate([self._X_reservoir, x[:take]], axis=0)
            self._y_reservoir = np.concatenate([self._y_reservoir, y[:take]], axis=0)
        else:
            for i in range(n):
                j = self._rng.randint(0, self._n_seen + i + 1)
                if j < capacity:
                    self._X_reservoir[j] = x[i]
                    self._y_reservoir[j] = y[i]
        self._n_seen += n

    def finalize_fit(self):
        if not self.enabled or self.max_weight == 0.0:
            return
        if self._X_reservoir is None or len(self._y_reservoir) < 1000:
            return
        params = dict(
            boosting_type="gbdt",
            objective="regression",
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            subsample_freq=1,
            colsample_bytree=self.colsample_bytree,
            min_child_samples=self.min_child_samples,
            reg_lambda=self.reg_lambda,
            verbose=-1,
            random_state=42,
            n_jobs=1,
        )
        self._weight = self._fit_weight(params)
        train_data = lgb.Dataset(self._X_reservoir, label=self._y_reservoir, free_raw_data=False)
        self._model = lgb.train(params, train_data, num_boost_round=self.n_estimators)

    def predict(self, raw: pd.DataFrame) -> np.ndarray:
        if not self.enabled or self._weight == 0.0 or self._model is None:
            return np.zeros(len(raw), dtype=np.float64)
        feats = self._transform_raw(raw)
        x = feats.loc[:, self._feature_names].to_numpy(dtype=np.float32, copy=False)
        return self._weight * self._model.predict(x).astype(np.float64)

    def _fit_weight(self, params):
        n = len(self._y_reservoir)
        calib_rows = min(max(int(n * self.calib_frac), self.min_calib_rows), n // 2)
        if calib_rows < 1000:
            return self.max_weight
        perm = self._rng.permutation(n)
        calib_idx = perm[:calib_rows]
        train_idx = perm[calib_rows:]
        if len(train_idx) < 1000:
            return self.max_weight
        train_data = lgb.Dataset(self._X_reservoir[train_idx], label=self._y_reservoir[train_idx], free_raw_data=False)
        calib_model = lgb.train(params, train_data, num_boost_round=self.n_estimators)
        pred = calib_model.predict(self._X_reservoir[calib_idx]).astype(np.float64)
        target = self._y_reservoir[calib_idx].astype(np.float64)
        denom = float(pred @ pred)
        if denom <= 1e-12:
            return self.max_weight
        beta = float((pred @ target) / denom)
        return float(np.clip(beta, 0.0, self.max_weight))
