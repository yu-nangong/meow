"""
MEOW workshop template: Ridge regression on 6 hand-crafted features.
Agents may replace this with deeper models in models/ or extend training here.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data_io import iter_days, train_test_dates, verify_data_dir
from feat import MeowFeatureGenerator
from mdl import MeowModel
from models.interval_residual import IntervalResidualRidge
from models.elasticnet_model import ElasticNetModel
from models.lgb_model import LGBModel
from models.panel_seasonality import PanelSeasonalityResidual
from models.blend_model import BlendModel
from models.lag_mlp_sequence_model import LagMLPSequenceModel
from models.deeplob_model import DeepLOBModel
from models.nn_residual import NnResidualModel

from models.pearson_nn import PearsonNNModel
MODEL_TYPE = os.environ.get("MEOW_MODEL_TYPE", "blend").strip().lower()
TRAIN_ON_INTERVAL_DEMEANED_TARGET = os.environ.get("MEOW_TRAIN_ON_INTERVAL_DEMEANED_TARGET", "0") != "0"

N_CHUNKS = int(os.environ.get("MEOW_N_CHUNKS", "8"))
FORECAST_CS_MEAN_SHRINK = float(os.environ.get("MEOW_FORECAST_CS_MEAN_SHRINK", "0.0"))
LEARN_FORECAST_CS_MEAN_SHRINK = os.environ.get("MEOW_LEARN_FORECAST_CS_MEAN_SHRINK", "0") != "0"
FORECAST_CS_MEAN_SHRINK_TAIL_DAYS = int(os.environ.get("MEOW_FORECAST_CS_MEAN_SHRINK_TAIL_DAYS", "10"))
FORECAST_CS_MEAN_SHRINK_MAX = float(os.environ.get("MEOW_FORECAST_CS_MEAN_SHRINK_MAX", "1.0"))
FORECAST_CS_MEAN_ADAPTIVE_BETA = float(os.environ.get("MEOW_FORECAST_CS_MEAN_ADAPTIVE_BETA", "0.0"))
SEQ_MODEL_ENABLED = os.environ.get("MEOW_SEQ_MODEL", "0") != "0"
SEQ_BLEND_WEIGHT = float(os.environ.get("MEOW_SEQ_BLEND_WEIGHT", "0.3"))
SEQ_MODEL_TYPE = os.environ.get("MEOW_SEQ_MODEL_TYPE", "mlp").strip().lower()
NN_RESIDUAL_ENABLED = os.environ.get("MEOW_NN_RESIDUAL", "0") != "0"
NN_RESIDUAL_BLEND_WEIGHT = float(os.environ.get("MEOW_NN_RESIDUAL_BLEND", "0.25"))
TARGET_STANDARDIZE = os.environ.get("MEOW_TARGET_STANDARDIZE", "1") != "0"
TARGET_STANDARDIZE_MIN_STD = 1e-8
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


def _compute_sym_target_stats(h5dir: str, train_dates: List[int]) -> tuple:
    """Compute per-symbol mean/std of fret12 across all training data."""
    sym_n = {}
    sym_sum = {}
    sym_sq = {}
    for day in train_dates:
        path = os.path.join(h5dir, f"{day}.h5")
        df = pd.read_hdf(path, columns=["symbol", "fret12"])
        for sym, grp in df.groupby("symbol", sort=False):
            y = grp["fret12"].to_numpy(dtype=np.float64)
            y = y[np.isfinite(y)]
            if len(y) == 0:
                continue
            sym_n[sym] = sym_n.get(sym, 0) + len(y)
            sym_sum[sym] = sym_sum.get(sym, 0.0) + float(y.sum())
            sym_sq[sym] = sym_sq.get(sym, 0.0) + float((y * y).sum())
        del df
    means = {}
    stds = {}
    for sym in sym_n:
        cnt = sym_n[sym]
        mean = sym_sum[sym] / cnt
        var = max(sym_sq[sym] / cnt - mean * mean, TARGET_STANDARDIZE_MIN_STD)
        means[sym] = mean
        stds[sym] = float(np.sqrt(var))
    return means, stds


def _zscore_target(ydf: pd.DataFrame, means: dict, stds: dict) -> pd.DataFrame:
    """Return ydf with fret12 replaced by per-symbol z-score."""
    ydf = ydf.copy()
    syms = ydf.index.get_level_values("symbol")
    y = ydf["fret12"].to_numpy(dtype=np.float64, copy=False)
    mean_arr = np.array([means.get(s, 0.0) for s in syms], dtype=np.float64)
    std_arr = np.array([stds.get(s, 1.0) for s in syms], dtype=np.float64)
    ydf["fret12"] = (y - mean_arr) / std_arr
    return ydf


def _unstd_preds(pred: np.ndarray, syms: np.ndarray, means: dict, stds: dict) -> np.ndarray:
    """Un-standardize predictions: pred_z * std_sym + mean_sym."""
    mean_arr = np.array([means.get(s, 0.0) for s in syms], dtype=np.float64)
    std_arr = np.array([stds.get(s, 1.0) for s in syms], dtype=np.float64)
    return pred.astype(np.float64) * std_arr + mean_arr


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


def _train_target_array(ydf: pd.DataFrame) -> np.ndarray:
    target = ydf["fret12"].to_numpy(dtype=np.float64, copy=False)
    if not TRAIN_ON_INTERVAL_DEMEANED_TARGET:
        return target
    frame = pd.DataFrame(
        {
            "date": ydf.index.get_level_values("date"),
            "interval": ydf.index.get_level_values("interval"),
            "fret12": target,
        }
    )
    group_mean = frame.groupby(["date", "interval"], sort=False)["fret12"].transform("mean")
    return target - group_mean.to_numpy(dtype=np.float64, copy=False)


def _postprocess_forecast(ydf: pd.DataFrame, pred: np.ndarray, mean_shrink: float) -> np.ndarray:
    if not mean_shrink and not FORECAST_CS_MEAN_ADAPTIVE_BETA:
        return pred
    out = _group_forecast_stats(ydf, pred)
    if FORECAST_CS_CENTER_STAT == "median":
        group_center = out["group_median"].to_numpy(dtype=np.float64, copy=False)
    else:
        group_center = out["group_mean"].to_numpy(dtype=np.float64, copy=False)
    shrink = mean_shrink
    if FORECAST_CS_MEAN_ADAPTIVE_BETA:
        mean_abs = np.abs(group_center)
        group_std = out["group_std"].to_numpy(dtype=np.float64, copy=False)
        common_mode_share = mean_abs / (mean_abs + group_std + 1e-12)
        shrink = np.clip(
            mean_shrink + FORECAST_CS_MEAN_ADAPTIVE_BETA * common_mode_share,
            0.0,
            FORECAST_CS_MEAN_SHRINK_MAX,
        )
    else:
        shrink = mean_shrink
    return pred - shrink * group_center


def _fit_forecast_mean_shrink(
    h5dir: str,
    feat_gen: MeowFeatureGenerator,
    model: MeowModel,
    train_dates: List[int],
) -> float:
    if FORECAST_CS_MEAN_SHRINK_TAIL_DAYS > 0:
        calib_dates = train_dates[-FORECAST_CS_MEAN_SHRINK_TAIL_DAYS :]
    else:
        calib_dates = train_dates
    numer = 0.0
    denom = 0.0
    for chunk in _chunk_dates(calib_dates, N_CHUNKS):
        raw = pd.concat(list(iter_days(h5dir, chunk)), ignore_index=True)
        xdf, ydf = feat_gen.genFeatures(raw)
        del raw
        pred = model.predict(xdf)
        del xdf
        group_mean = _group_forecast_stats(ydf, pred)["group_mean"].to_numpy(dtype=np.float64, copy=False)
        err = pred - ydf["fret12"].to_numpy(dtype=np.float64, copy=False)
        numer += float(err @ group_mean)
        denom += float(group_mean @ group_mean)
    if denom <= 0.0:
        return FORECAST_CS_MEAN_SHRINK
    return float(np.clip(numer / denom, 0.0, FORECAST_CS_MEAN_SHRINK_MAX))


def _create_base_model():
    if MODEL_TYPE == "blend":
        return BlendModel()
    if MODEL_TYPE == "lgb":
        return LGBModel()
    if MODEL_TYPE == "pearson_nn":
        return PearsonNNModel()
    if MODEL_TYPE == "elasticnet":
        return ElasticNetModel(cacheDir=None)
    return MeowModel(cacheDir=None)


def train_and_evaluate(h5dir: Optional[str] = None) -> Dict[str, float]:
    h5dir = _resolve_h5dir(h5dir)
    train_dates, test_dates = train_test_dates()
    feat_gen = MeowFeatureGenerator(cacheDir=None)
    model = _create_base_model()
    model.reset()
    sym_means = {}
    sym_stds = {}
    if TARGET_STANDARDIZE:
        sym_means, sym_stds = _compute_sym_target_stats(h5dir, train_dates)

    for chunk in _chunk_dates(train_dates, N_CHUNKS):
        raw = pd.concat(list(iter_days(h5dir, chunk)), ignore_index=True)
        xdf, ydf = feat_gen.genFeatures(raw)
        del raw
        if TARGET_STANDARDIZE:
            ydf = _zscore_target(ydf, sym_means, sym_stds)
        y_train = ydf.copy()
        y_train.loc[:, "fret12"] = _train_target_array(ydf)
        model.partial_fit(xdf, y_train)
        del xdf, ydf, y_train
    model.finalize_fit()
    forecast_cs_mean_shrink = FORECAST_CS_MEAN_SHRINK
    if LEARN_FORECAST_CS_MEAN_SHRINK:
        forecast_cs_mean_shrink = _fit_forecast_mean_shrink(h5dir, feat_gen, model, train_dates)
    interval_residual = IntervalResidualRidge()
    for chunk in _chunk_dates(train_dates, N_CHUNKS):
        raw = pd.concat(list(iter_days(h5dir, chunk)), ignore_index=True)
        xdf, ydf = feat_gen.genFeatures(raw)
        del raw
        if TARGET_STANDARDIZE:
            ydf = _zscore_target(ydf, sym_means, sym_stds)
        base_pred = _postprocess_forecast(ydf, model.predict(xdf), forecast_cs_mean_shrink)
        resid = _train_target_array(ydf) - base_pred
        interval_residual.partial_fit(xdf, resid, base_pred=base_pred)
        del xdf, ydf, base_pred, resid
    interval_residual.finalize_fit()
    panel_residual = PanelSeasonalityResidual()
    if panel_residual.enabled and panel_residual.blend:
        panel_dates = train_dates[-panel_residual.tail_days :] if panel_residual.tail_days > 0 else train_dates
        for chunk in _chunk_dates(panel_dates, N_CHUNKS):
            raw = pd.concat(list(iter_days(h5dir, chunk)), ignore_index=True)
            xdf, ydf = feat_gen.genFeatures(raw)
            del raw
            if TARGET_STANDARDIZE:
                ydf = _zscore_target(ydf, sym_means, sym_stds)
            forecast = _postprocess_forecast(ydf, model.predict(xdf), forecast_cs_mean_shrink)
            forecast = forecast + interval_residual.predict(xdf, base_pred=forecast)
            resid = ydf["fret12"].to_numpy(dtype=np.float64, copy=False) - forecast
            panel_residual.partial_fit(ydf, resid)
            del xdf, ydf, forecast, resid
        panel_residual.finalize_fit()
    nn_residual = None
    if NN_RESIDUAL_ENABLED:
        nn_residual = NnResidualModel()
        nn_residual.reset()
        for chunk in _chunk_dates(train_dates, N_CHUNKS):
            raw = pd.concat(list(iter_days(h5dir, chunk)), ignore_index=True)
            xdf, ydf = feat_gen.genFeatures(raw)
            del raw
            if TARGET_STANDARDIZE:
                ydf = _zscore_target(ydf, sym_means, sym_stds)
            forecast = _postprocess_forecast(ydf, model.predict(xdf), forecast_cs_mean_shrink)
            forecast = forecast + interval_residual.predict(xdf, base_pred=forecast)
            forecast = forecast + panel_residual.predict(ydf)
            resid = ydf["fret12"].to_numpy(dtype=np.float64, copy=False) - forecast
            nn_residual.partial_fit(xdf, resid)
            del xdf, ydf, forecast, resid
        nn_residual.finalize_fit()

    seq_model = None
    if SEQ_MODEL_ENABLED:
        if SEQ_MODEL_TYPE == "deeplob":
            seq_model = DeepLOBModel()
        else:
            seq_model = LagMLPSequenceModel()
        seq_model.reset()
        for chunk in _chunk_dates(train_dates, N_CHUNKS):
            raw = pd.concat(list(iter_days(h5dir, chunk)), ignore_index=True)
            # Train sequence model on raw fret12 (not residuals)
            seq_model.partial_fit(raw)
            del raw
        seq_model.finalize_fit()


    y_parts, p_parts = [], []
    for chunk in _chunk_dates(test_dates, N_CHUNKS):
        raw = pd.concat(list(iter_days(h5dir, chunk)), ignore_index=True)
        raw = raw.sort_values(["symbol", "date", "interval"], kind="mergesort")
        if seq_model is not None:
            seq_pred = seq_model.predict(raw)
        else:
            seq_pred = np.zeros(len(raw), dtype=np.float64)
        xdf, ydf = feat_gen.genFeatures(raw)
        ydf = ydf.copy()
        forecast = model.predict(xdf)
        forecast = forecast + interval_residual.predict(xdf, base_pred=forecast)
        forecast = forecast + panel_residual.predict(ydf)
        if nn_residual is not None:
            nn_resid = nn_residual.predict(xdf)
        else:
            nn_resid = np.zeros(len(raw), dtype=np.float64)
        forecast = forecast + NN_RESIDUAL_BLEND_WEIGHT * nn_resid
        forecast = (1.0 - SEQ_BLEND_WEIGHT) * forecast + SEQ_BLEND_WEIGHT * seq_pred
        if TARGET_STANDARDIZE:
            syms = ydf.index.get_level_values("symbol").to_numpy()
            forecast = _unstd_preds(forecast, syms, sym_means, sym_stds)
        forecast = _postprocess_forecast(ydf, forecast, forecast_cs_mean_shrink)
        ydf.loc[:, "forecast"] = forecast
        del raw, xdf
        y_parts.append(ydf["fret12"].to_numpy())
        p_parts.append(ydf["forecast"].to_numpy())

    return _pearson_metrics(np.concatenate(y_parts), np.concatenate(p_parts))
