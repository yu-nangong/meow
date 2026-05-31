"""
Online NN replacing 2-stage (base + interval residual).

Single hidden layer neural network trained via online SGD.
No data accumulation — processes chunks incrementally.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data_io import iter_days, train_test_dates, verify_data_dir
from feat import MeowFeatureGenerator
from models.online_nn import OnlineNN

N_CHUNKS = int(os.environ.get("MEOW_N_CHUNKS", "8"))


def _chunk_dates(dates: List[int], n_chunks: int) -> List[List[int]]:
    if not dates:
        return []
    n_chunks = min(n_chunks, len(dates))
    size = (len(dates) + n_chunks - 1) // n_chunks
    return [dates[i : i + size] for i in range(0, len(dates), size)]


def _pearson_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return {"pearson": 0.0, "r2": 0.0, "mse": 0.0}
    pcor = float(np.corrcoef(y_true, y_pred)[0, 1])
    mse = float(np.mean((y_pred - y_true) ** 2))
    var = float(np.var(y_true))
    r2 = float(1 - mse / var) if var > 0 else 0.0
    return {"pearson": pcor, "r2": r2, "mse": mse}


def _resolve_h5dir(h5dir: Optional[str]) -> str:
    if h5dir:
        return verify_data_dir(h5dir)
    return verify_data_dir()


def train_and_evaluate(h5dir: Optional[str] = None) -> Dict[str, float]:
    h5dir = _resolve_h5dir(h5dir)
    train_dates, test_dates = train_test_dates()
    feat_gen = MeowFeatureGenerator(cacheDir=None)
    model = OnlineNN()
    model.reset()

    for chunk in _chunk_dates(train_dates, N_CHUNKS):
        raw = pd.concat(list(iter_days(h5dir, chunk)), ignore_index=True)
        xdf, ydf = feat_gen.genFeatures(raw)
        del raw
        model.partial_fit(xdf, ydf)
        del xdf, ydf
    model.finalize_fit()

    y_parts, p_parts = [], []
    for chunk in _chunk_dates(test_dates, N_CHUNKS):
        raw = pd.concat(list(iter_days(h5dir, chunk)), ignore_index=True)
        xdf, ydf = feat_gen.genFeatures(raw)
        del raw
        ydf = ydf.copy()
        forecast = model.predict(xdf)
        ydf.loc[:, "forecast"] = forecast
        del xdf
        y_parts.append(ydf["fret12"].to_numpy())
        p_parts.append(ydf["forecast"].to_numpy())

    return _pearson_metrics(np.concatenate(y_parts), np.concatenate(p_parts))
