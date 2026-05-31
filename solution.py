"""
MEOW solution: Incremental LightGBM with interval/time features directly.
No two-stage interval-residual correction.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data_io import iter_days, train_test_dates, verify_data_dir
from feat import MeowFeatureGenerator

from models.lgbm_incremental import LightGBMIncremental

N_CHUNKS = int(os.environ.get("MEOW_N_CHUNKS", "8"))
FORECAST_CS_MEAN_SHRINK = float(os.environ.get("MEOW_FORECAST_CS_MEAN_SHRINK", "0.25"))
FORECAST_CS_MEAN_SHRINK_MAX = float(os.environ.get("MEOW_FORECAST_CS_MEAN_SHRINK_MAX", "1.0"))
FORECAST_CS_CENTER_STAT = os.environ.get("MEOW_FORECAST_CS_CENTER_STAT", "median").strip().lower()


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


def _group_forecast_stats(ydf: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": ydf.index.get_level_values("date"),
            "interval": ydf.index.get_level_values("interval"),
            "forecast": pred,
        }
    )
    grp = out.groupby(["date", "interval"], sort=False)["forecast"]
    out["group_mean"] = grp.transform("mean")
    out["group_median"] = grp.transform("median")
    out["group_std"] = grp.transform("std").fillna(0.0)
    return out


def _postprocess_forecast(ydf: pd.DataFrame, pred: np.ndarray, mean_shrink: float) -> np.ndarray:
    if not mean_shrink:
        return pred
    out = _group_forecast_stats(ydf, pred)
    if FORECAST_CS_CENTER_STAT == "median":
        group_center = out["group_median"].to_numpy(dtype=np.float64, copy=False)
    else:
        group_center = out["group_mean"].to_numpy(dtype=np.float64, copy=False)
    return pred - mean_shrink * group_center


def train_and_evaluate(h5dir: Optional[str] = None) -> Dict[str, float]:
    h5dir = _resolve_h5dir(h5dir)
    train_dates, test_dates = train_test_dates()
    feat_gen = MeowFeatureGenerator(cacheDir=None)
    model = LightGBMIncremental(cacheDir=None)
    model.reset()

    # Incremental training on chunks — never concatenates all data
    for chunk in _chunk_dates(train_dates, N_CHUNKS):
        raw = pd.concat(list(iter_days(h5dir, chunk)), ignore_index=True)
        xdf, ydf = feat_gen.genFeatures(raw)
        del raw
        model.partial_fit(xdf, ydf)
        del xdf, ydf
    model.finalize_fit()

    # Test
    y_parts, p_parts = [], []
    for chunk in _chunk_dates(test_dates, N_CHUNKS):
        raw = pd.concat(list(iter_days(h5dir, chunk)), ignore_index=True)
        xdf, ydf = feat_gen.genFeatures(raw)
        del raw
        forecast = _postprocess_forecast(ydf, model.predict(xdf), FORECAST_CS_MEAN_SHRINK)
        ydf.loc[:, "forecast"] = forecast
        del xdf
        y_parts.append(ydf["fret12"].to_numpy())
        p_parts.append(ydf["forecast"].to_numpy())

    return _pearson_metrics(np.concatenate(y_parts), np.concatenate(p_parts))
