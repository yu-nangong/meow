"""Budget-safe lag-window MLP arm on raw LOB snapshots."""
from __future__ import annotations

import os

import numpy as np
from sklearn.neural_network import MLPRegressor


class LagMLPSequenceModel:
    def __init__(self):
        self.lookback = int(os.environ.get("MEOW_SEQ_LOOKBACK", "8"))
        self.max_rows = int(os.environ.get("MEOW_SEQ_MAX_ROWS", "120000"))
        self.hidden = tuple(
            int(part.strip())
            for part in os.environ.get("MEOW_SEQ_HIDDEN", "48,24").split(",")
            if part.strip()
        )
        self.max_iter = int(os.environ.get("MEOW_SEQ_EPOCHS", "12"))
        self.alpha = float(os.environ.get("MEOW_SEQ_ALPHA", "1e-4"))
        self.learning_rate_init = float(os.environ.get("MEOW_SEQ_LR", "8e-4"))
        self.feature_cols = [
            col.strip()
            for col in os.environ.get(
                "MEOW_SEQ_COLS",
                "bid0,ask0,bsize0,asize0,tradeBuyQty,tradeSellQty,"
                "tradeBuyTurnover,tradeSellTurnover,midpx,lastpx",
            ).split(",")
            if col.strip()
        ]
        self._reservoir_x = None
        self._reservoir_y = None
        self._n_seen = 0
        self._mean = None
        self._scale = None
        self._model = None
        self._active_cols = None
        self._price_idx = None
        self._mid_idx = None
        self._rng = np.random.RandomState(42)

    def reset(self):
        self._reservoir_x = None
        self._reservoir_y = None
        self._n_seen = 0
        self._mean = None
        self._scale = None
        self._model = None
        self._active_cols = None
        self._price_idx = None
        self._mid_idx = None

    def partial_fit(self, raw):
        seq_x, seq_y = self._build_train_examples(raw)
        if len(seq_y) == 0:
            return
        n = len(seq_y)
        if self._reservoir_x is None:
            take = min(n, self.max_rows)
            self._reservoir_x = seq_x[:take].copy()
            self._reservoir_y = seq_y[:take].copy()
            self._n_seen = n
            return
        capacity = self._reservoir_x.shape[0]
        if capacity < self.max_rows:
            take = min(n, self.max_rows - capacity)
            self._reservoir_x = np.concatenate([self._reservoir_x, seq_x[:take]], axis=0)
            self._reservoir_y = np.concatenate([self._reservoir_y, seq_y[:take]], axis=0)
            start = take
        else:
            start = 0
        for i in range(start, n):
            j = self._rng.randint(0, self._n_seen + i + 1)
            if j < self.max_rows:
                self._reservoir_x[j] = seq_x[i]
                self._reservoir_y[j] = seq_y[i]
        self._n_seen += n

    def finalize_fit(self):
        if self._reservoir_y is None or len(self._reservoir_y) < 5000:
            return
        self._mean = self._reservoir_x.mean(axis=0, dtype=np.float64)
        self._scale = self._reservoir_x.std(axis=0, dtype=np.float64)
        self._scale = np.where(self._scale > 1e-6, self._scale, 1.0)
        x_train = ((self._reservoir_x - self._mean) / self._scale).astype(np.float32, copy=False)
        self._model = MLPRegressor(
            hidden_layer_sizes=self.hidden,
            activation="relu",
            solver="adam",
            alpha=self.alpha,
            batch_size=1024,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=3,
            shuffle=True,
            random_state=42,
        )
        self._model.fit(x_train, self._reservoir_y)

    def predict(self, raw):
        pred = np.zeros(len(raw), dtype=np.float64)
        if self._model is None:
            return pred
        seq_x, row_ids = self._build_predict_examples(raw)
        if len(row_ids) == 0:
            return pred
        seq_x = ((seq_x - self._mean) / self._scale).astype(np.float32, copy=False)
        pred[row_ids] = self._model.predict(seq_x).astype(np.float64, copy=False)
        return pred

    def _build_train_examples(self, raw):
        seq_x_parts = []
        seq_y_parts = []
        for norm_values, target, _row_ids in self._iter_group_windows(raw):
            windows = self._window_matrix(norm_values)
            if windows is None:
                continue
            seq_x_parts.append(windows)
            seq_y_parts.append(target[self.lookback - 1 :].astype(np.float32, copy=False))
        if not seq_x_parts:
            width = self.lookback * max(len(self._active_cols or self.feature_cols), 1)
            return (
                np.zeros((0, width), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )
        return (
            np.concatenate(seq_x_parts, axis=0).astype(np.float32, copy=False),
            np.concatenate(seq_y_parts, axis=0).astype(np.float32, copy=False),
        )

    def _build_predict_examples(self, raw):
        seq_x_parts = []
        row_id_parts = []
        for norm_values, _target, row_ids in self._iter_group_windows(raw):
            windows = self._window_matrix(norm_values)
            if windows is None:
                continue
            seq_x_parts.append(windows)
            row_id_parts.append(row_ids[self.lookback - 1 :])
        if not seq_x_parts:
            width = self.lookback * max(len(self._active_cols or self.feature_cols), 1)
            return np.zeros((0, width), dtype=np.float32), np.zeros(0, dtype=np.int64)
        return (
            np.concatenate(seq_x_parts, axis=0).astype(np.float32, copy=False),
            np.concatenate(row_id_parts, axis=0).astype(np.int64, copy=False),
        )

    def _iter_group_windows(self, raw):
        raw = raw.reset_index(drop=True)
        active = self._resolve_active_cols(raw)
        if not active:
            return
        work = raw.loc[:, ["date", "symbol", "interval", "fret12", *active]].copy()
        work["__row_id"] = np.arange(len(work), dtype=np.int64)
        work = work.sort_values(["date", "symbol", "interval"], kind="mergesort")
        group_cols = ["date", "symbol"]
        for _, gdf in work.groupby(group_cols, sort=False):
            if len(gdf) < self.lookback:
                continue
            values = gdf.loc[:, active].to_numpy(dtype=np.float64, copy=False)
            target = gdf["fret12"].to_numpy(dtype=np.float64, copy=False)
            row_ids = gdf["__row_id"].to_numpy(dtype=np.int64, copy=False)
            yield self._normalize_group(values), target, row_ids

    def _resolve_active_cols(self, raw):
        if self._active_cols is None:
            self._active_cols = [col for col in self.feature_cols if col in raw.columns]
            self._price_idx = [
                idx
                for idx, col in enumerate(self._active_cols)
                if col.startswith("bid")
                or col.startswith("ask")
                or col in {"midpx", "lastpx"}
            ]
            self._mid_idx = self._active_cols.index("midpx") if "midpx" in self._active_cols else None
        return self._active_cols

    def _normalize_group(self, values):
        values = np.nan_to_num(values.astype(np.float64, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
        if self._mid_idx is not None:
            mid = np.maximum(np.abs(values[:, self._mid_idx : self._mid_idx + 1]), 1e-6)
            if self._price_idx:
                values[:, self._price_idx] = values[:, self._price_idx] / mid - 1.0
        non_price = np.ones(values.shape[1], dtype=bool)
        if self._price_idx:
            non_price[self._price_idx] = False
        values[:, non_price] = np.sign(values[:, non_price]) * np.log1p(np.abs(values[:, non_price]))
        return values.astype(np.float32, copy=False)

    def _window_matrix(self, norm_values):
        n_rows, n_features = norm_values.shape
        if n_rows < self.lookback:
            return None
        windows = np.lib.stride_tricks.sliding_window_view(norm_values, (self.lookback, n_features))
        windows = windows[:, 0].reshape(n_rows - self.lookback + 1, self.lookback * n_features)
        return windows.astype(np.float32, copy=False)
