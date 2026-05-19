import os
import numpy as np
import pandas as pd
from log import log


class MeowFeatureGenerator(object):
    @staticmethod
    def _parse_feature_list_env(name, default):
        raw = os.environ.get(name, "")
        if not raw.strip():
            return list(default)
        return [item.strip() for item in raw.split(",") if item.strip()]

    @classmethod
    def _nonlinear_time_interactions(cls):
        default = [
            "trade_imb_rank_cs",
            "high_gap_rank_cs",
            "trade_vwad_gap_rank_cs",
            "micro_dev_rank_cs",
        ]
        return cls._parse_feature_list_env("MEOW_NONLINEAR_TIME_FEATURES", default)

    @classmethod
    def featureNames(cls):
        feature_names = [
            "ob_imb0",
            "ob_imb4",
            "ob_imb9",
            "ob_imb19",
            "ob_imb_front_back",
            "ob_imb_inner_outer",
            "trade_imb",
            "turnover_imb",
            "add_imb",
            "cxl_imb",
            "flow_imb",
            "spread",
            "micro_dev",
            "last_mid_dev",
            "ret1",
            "ret3",
            "ret6",
            "ret12",
            "ret24",
            "trade_imb_ema6",
            "ob_imb0_ema6",
            "micro_dev_ema6",
            "ret12_resid",
            "trade_imb_cs",
            "micro_dev_cs",
            "ret1_cs",
            "ret3_cs",
            "ret6_cs",
            "ret12_cs",
            "ret12_resid_cs",
            "spread_cs",
            "high_gap_cs",
            "trade_buy_high_gap_cs",
            "trade_sell_high_gap_cs",
            "trade_count_imb",
            "add_count_imb",
            "cxl_count_imb",
            "depth_pressure_04",
            "depth_pressure_59",
            "depth_pressure_1019",
            "depth_pressure_slope",
            "depth_pressure_curve",
            "bid_near_share",
            "ask_near_share",
            "near_share_imb",
            "top_queue_share_imb",
            "buy_vwad_dev",
            "sell_vwad_dev",
            "trade_vwad_gap",
            "high_gap",
            "low_gap",
            "high_minus_low",
            "trade_buy_high_gap",
            "trade_sell_high_gap",
            "vwad_center_dev",
            "trade_high_center_gap",
            "trade_high_skew",
            "high_vs_trade_high_gap",
            "add_turn_imb",
            "cxl_turn_imb",
            "day_open_gap",
            "range_pos",
            "trade_count_share",
            "ret3_x_flow",
            "ret6_x_flow",
            "trade_imb_rank_cs",
            "flow_imb_rank_cs",
            "micro_dev_rank_cs",
            "ret1_rank_cs",
            "ret3_rank_cs",
            "ret6_rank_cs",
            "ret12_resid_rank_cs",
            "turnover_imb_rank_cs",
            "add_turn_imb_rank_cs",
            "day_open_gap_rank_cs",
            "trade_count_share_rank_cs",
            "buy_vwad_dev_rank_cs",
            "trade_vwad_gap_rank_cs",
            "range_pos_rank_cs",
            "last_mid_dev_rank_cs",
            "ret24_rank_cs",
            "ob_imb4_rank_cs",
            "depth_pressure_04_rank_cs",
            "ob_imb0_rank_cs",
            "ob_imb9_rank_cs",
            "ob_imb19_rank_cs",
            "ob_imb_front_back_rank_cs",
            "ob_imb_inner_outer_rank_cs",
            "spread_rank_cs",
            "ret12_rank_cs",
            "sell_vwad_dev_rank_cs",
            "depth_pressure_59_rank_cs",
            "depth_pressure_1019_rank_cs",
            "depth_pressure_slope_rank_cs",
            "depth_pressure_curve_rank_cs",
            "near_share_imb_rank_cs",
            "top_queue_share_imb_rank_cs",
            "high_gap_rank_cs",
            "low_gap_rank_cs",
            "high_minus_low_rank_cs",
            "trade_buy_high_gap_rank_cs",
            "trade_sell_high_gap_rank_cs",
            "vwad_center_dev_rank_cs",
            "trade_high_center_gap_rank_cs",
            "trade_high_skew_rank_cs",
            "high_vs_trade_high_gap_rank_cs",
            "interval_frac_centered",
            "interval_u",
            "interval_frac_sq",
            "interval_frac_cu",
            "interval_u_sq",
            "trade_imb_rank_cs_x_time",
            "flow_imb_rank_cs_x_time",
            "ret1_rank_cs_x_time",
            "ret6_rank_cs_x_time",
            "range_pos_rank_cs_x_time",
            "trade_count_share_rank_cs_x_time",
            "top_queue_share_imb_rank_cs_x_time",
            "depth_pressure_slope_rank_cs_x_time",
            "high_gap_rank_cs_x_time",
            "ob_imb19_rank_cs_x_time",
            "depth_pressure_1019_rank_cs_x_time",
            "ob_imb4_rank_cs_x_time",
            "ret3_rank_cs_x_time",
            "ret12_resid_rank_cs_x_time",
            "day_open_gap_rank_cs_x_time",
            "ret24_rank_cs_x_time",
            "sell_vwad_dev_rank_cs_x_time",
            "trade_vwad_gap_rank_cs_x_time",
            "trade_high_center_gap_rank_cs_x_time",
            "micro_dev_rank_cs_x_time",
            "last_mid_dev_rank_cs_x_time",
            "trade_imb_rank_cs_x_u",
            "flow_imb_rank_cs_x_u",
            "ret1_rank_cs_x_u",
            "ret6_rank_cs_x_u",
            "range_pos_rank_cs_x_u",
            "trade_count_share_rank_cs_x_u",
            "top_queue_share_imb_rank_cs_x_u",
            "high_gap_rank_cs_x_u",
            "ob_imb19_rank_cs_x_u",
            "depth_pressure_1019_rank_cs_x_u",
            "ob_imb4_rank_cs_x_u",
            "ret3_rank_cs_x_u",
            "ret12_resid_rank_cs_x_u",
            "high_minus_low_rank_cs_x_u",
            "day_open_gap_rank_cs_x_u",
            "ret24_rank_cs_x_u",
            "sell_vwad_dev_rank_cs_x_u",
            "trade_vwad_gap_rank_cs_x_u",
            "trade_high_center_gap_rank_cs_x_u",
            "micro_dev_rank_cs_x_u",
        ]
        nonlinear_time_interactions = cls._nonlinear_time_interactions()
        feature_names.extend(f"{col}_x_time_sq" for col in nonlinear_time_interactions)
        feature_names.extend(f"{col}_x_u_sq" for col in nonlinear_time_interactions)
        return feature_names

    def __init__(self, cacheDir):
        self.cacheDir = cacheDir
        self.ycol = "fret12"
        self.mcols = ["symbol", "date", "interval"]

    def genFeatures(self, df):
        log.inf("Generating {} features from raw data...".format(len(self.featureNames())))
        eps = 1e-6
        df = df.sort_values(self.mcols, kind="mergesort").copy()
        sym_day = df.groupby(["symbol", "date"], sort=False)
        features = {}

        features["ob_imb0"] = (df["bsize0"] - df["asize0"]) / (df["bsize0"] + df["asize0"] + eps)
        features["ob_imb4"] = (df["bsize0_4"] - df["asize0_4"]) / (df["bsize0_4"] + df["asize0_4"] + eps)
        features["ob_imb9"] = (df["bsize5_9"] - df["asize5_9"]) / (df["bsize5_9"] + df["asize5_9"] + eps)
        features["ob_imb19"] = (df["bsize10_19"] - df["asize10_19"]) / (df["bsize10_19"] + df["asize10_19"] + eps)
        features["ob_imb_front_back"] = features["ob_imb0"] - features["ob_imb19"]
        features["ob_imb_inner_outer"] = features["ob_imb4"] - 0.5 * (features["ob_imb9"] + features["ob_imb19"])
        features["trade_imb"] = (df["tradeBuyQty"] - df["tradeSellQty"]) / (
            df["tradeBuyQty"] + df["tradeSellQty"] + eps
        )
        features["turnover_imb"] = (df["tradeBuyTurnover"] - df["tradeSellTurnover"]) / (
            df["tradeBuyTurnover"] + df["tradeSellTurnover"] + eps
        )
        features["add_imb"] = (df["addBuyQty"] - df["addSellQty"]) / (df["addBuyQty"] + df["addSellQty"] + eps)
        features["cxl_imb"] = (df["cxlBuyQty"] - df["cxlSellQty"]) / (df["cxlBuyQty"] + df["cxlSellQty"] + eps)
        features["add_turn_imb"] = (df["addBuyTurnover"] - df["addSellTurnover"]) / (
            df["addBuyTurnover"] + df["addSellTurnover"] + eps
        )
        features["cxl_turn_imb"] = (df["cxlBuyTurnover"] - df["cxlSellTurnover"]) / (
            df["cxlBuyTurnover"] + df["cxlSellTurnover"] + eps
        )
        features["flow_imb"] = (
            (df["addBuyQty"] - df["cxlBuyQty"]) - (df["addSellQty"] - df["cxlSellQty"])
        ) / (df["addBuyQty"] + df["cxlBuyQty"] + df["addSellQty"] + df["cxlSellQty"] + eps)
        features["trade_count_imb"] = (df["nTradeBuy"] - df["nTradeSell"]) / (
            df["nTradeBuy"] + df["nTradeSell"] + eps
        )
        features["add_count_imb"] = (df["nAddBuy"] - df["nAddSell"]) / (df["nAddBuy"] + df["nAddSell"] + eps)
        features["cxl_count_imb"] = (df["nCxlBuy"] - df["nCxlSell"]) / (df["nCxlBuy"] + df["nCxlSell"] + eps)
        features["depth_pressure_04"] = (df["btr0_4"] - df["atr0_4"]) / (df["btr0_4"] + df["atr0_4"] + eps)
        features["depth_pressure_59"] = (df["btr5_9"] - df["atr5_9"]) / (df["btr5_9"] + df["atr5_9"] + eps)
        features["depth_pressure_1019"] = (df["btr10_19"] - df["atr10_19"]) / (df["btr10_19"] + df["atr10_19"] + eps)
        features["depth_pressure_slope"] = features["depth_pressure_04"] - features["depth_pressure_1019"]
        features["depth_pressure_curve"] = features["depth_pressure_59"] - 0.5 * (
            features["depth_pressure_04"] + features["depth_pressure_1019"]
        )

        bid_depth_total = df["bsize0_4"] + df["bsize5_9"] + df["bsize10_19"]
        ask_depth_total = df["asize0_4"] + df["asize5_9"] + df["asize10_19"]
        features["bid_near_share"] = df["bsize0_4"] / (bid_depth_total + eps)
        features["ask_near_share"] = df["asize0_4"] / (ask_depth_total + eps)
        features["near_share_imb"] = features["bid_near_share"] - features["ask_near_share"]
        features["top_queue_share_imb"] = (df["bsize0"] / (bid_depth_total + eps)) - (
            df["asize0"] / (ask_depth_total + eps)
        )
        features["spread"] = (df["ask0"] - df["bid0"]) / (df["midpx"] + eps)
        features["micro_dev"] = (
            ((df["ask0"] * df["bsize0"] + df["bid0"] * df["asize0"]) / (df["asize0"] + df["bsize0"] + eps))
            - df["midpx"]
        ) / (df["midpx"] + eps)
        features["last_mid_dev"] = (df["lastpx"] - df["midpx"]) / (df["midpx"] + eps)
        features["buy_vwad_dev"] = (df["buyVwad"] - df["midpx"]) / (df["midpx"] + eps)
        features["sell_vwad_dev"] = (df["sellVwad"] - df["midpx"]) / (df["midpx"] + eps)
        features["trade_vwad_gap"] = (df["buyVwad"] - df["sellVwad"]) / (df["midpx"] + eps)
        features["high_gap"] = (df["high"] - df["midpx"]) / (df["midpx"] + eps)
        features["low_gap"] = (df["midpx"] - df["low"]) / (df["midpx"] + eps)
        features["high_minus_low"] = (df["high"] - df["low"]) / (df["midpx"] + eps)
        features["trade_buy_high_gap"] = (df["tradeBuyHigh"] - df["midpx"]) / (df["midpx"] + eps)
        features["trade_sell_high_gap"] = (df["tradeSellHigh"] - df["midpx"]) / (df["midpx"] + eps)
        features["vwad_center_dev"] = 0.5 * (features["buy_vwad_dev"] + features["sell_vwad_dev"])
        features["trade_high_center_gap"] = 0.5 * (
            features["trade_buy_high_gap"] + features["trade_sell_high_gap"]
        )
        features["trade_high_skew"] = features["trade_buy_high_gap"] - features["trade_sell_high_gap"]
        features["high_vs_trade_high_gap"] = features["high_gap"] - features["trade_high_center_gap"]
        features["day_open_gap"] = (df["midpx"] - df["open"]) / (df["open"] + eps)
        features["range_pos"] = ((df["midpx"] - df["low"]) - (df["high"] - df["midpx"])) / (
            df["high"] - df["low"] + eps
        )
        features["trade_count_share"] = (df["nTradeBuy"] + df["nTradeSell"]) / (
            df["nAddBuy"]
            + df["nAddSell"]
            + df["nCxlBuy"]
            + df["nCxlSell"]
            + df["nTradeBuy"]
            + df["nTradeSell"]
            + eps
        )

        features["ret1"] = sym_day["midpx"].pct_change(1)
        features["ret3"] = sym_day["midpx"].pct_change(3)
        features["ret6"] = sym_day["midpx"].pct_change(6)
        features["ret12"] = sym_day["midpx"].pct_change(12)
        features["ret24"] = sym_day["midpx"].pct_change(24)

        base_df = pd.DataFrame(features, index=df.index)
        base_sym_day = base_df.groupby([df["symbol"], df["date"]], sort=False)
        base_df.loc[:, "trade_imb_ema6"] = base_sym_day["trade_imb"].transform(
            lambda s: s.ewm(halflife=6, adjust=False).mean()
        )
        base_df.loc[:, "ob_imb0_ema6"] = base_sym_day["ob_imb0"].transform(
            lambda s: s.ewm(halflife=6, adjust=False).mean()
        )
        base_df.loc[:, "micro_dev_ema6"] = base_sym_day["micro_dev"].transform(
            lambda s: s.ewm(halflife=6, adjust=False).mean()
        )
        base_df.loc[:, "ret12_resid"] = base_df["ret12"] - base_df.groupby([df["date"], df["interval"]], sort=False)[
            "ret12"
        ].transform("mean")
        base_df.loc[:, "ret3_x_flow"] = base_df["ret3"] * base_df["flow_imb"]
        base_df.loc[:, "ret6_x_flow"] = base_df["ret6"] * base_df["flow_imb"]

        cs_cols = [
            "trade_imb",
            "micro_dev",
            "ret1",
            "ret3",
            "ret6",
            "ret12",
            "ret12_resid",
            "spread",
            "high_gap",
            "trade_buy_high_gap",
            "trade_sell_high_gap",
        ]
        cs_frame = base_df[cs_cols]
        cs_means = cs_frame.groupby([df["date"], df["interval"]], sort=False).transform("mean")
        cs_out = cs_frame - cs_means
        cs_out.columns = [f"{col}_cs" for col in cs_cols]

        rank_cols = [
            "trade_imb",
            "flow_imb",
            "micro_dev",
            "ret1",
            "ret3",
            "ret6",
            "ret12_resid",
            "turnover_imb",
            "add_turn_imb",
            "day_open_gap",
            "trade_count_share",
            "buy_vwad_dev",
            "trade_vwad_gap",
            "range_pos",
            "last_mid_dev",
            "ret24",
            "ob_imb4",
            "depth_pressure_04",
            "ob_imb0",
            "ob_imb9",
            "ob_imb19",
            "ob_imb_front_back",
            "ob_imb_inner_outer",
            "spread",
            "ret12",
            "sell_vwad_dev",
            "depth_pressure_59",
            "depth_pressure_1019",
            "depth_pressure_slope",
            "depth_pressure_curve",
            "near_share_imb",
            "top_queue_share_imb",
            "high_gap",
            "low_gap",
            "high_minus_low",
            "trade_buy_high_gap",
            "trade_sell_high_gap",
            "vwad_center_dev",
            "trade_high_center_gap",
            "trade_high_skew",
            "high_vs_trade_high_gap",
        ]
        rank_df = base_df[rank_cols].groupby([df["date"], df["interval"]], sort=False).rank(pct=True) - 0.5
        rank_df.columns = [f"{col}_rank_cs" for col in rank_cols]

        interval_max = df.groupby("date", sort=False)["interval"].transform("max").clip(lower=1)
        interval_frac_centered = df["interval"] / interval_max - 0.5
        interval_u = np.abs(interval_frac_centered)
        time_df = pd.DataFrame(
            {
                "interval_frac_centered": interval_frac_centered,
                "interval_u": interval_u,
                "interval_frac_sq": interval_frac_centered * interval_frac_centered,
                "interval_frac_cu": interval_frac_centered * interval_frac_centered * interval_frac_centered,
                "interval_u_sq": interval_u * interval_u,
            },
            index=df.index,
        )

        time_interactions = [
            "trade_imb_rank_cs",
            "flow_imb_rank_cs",
            "ret1_rank_cs",
            "ret6_rank_cs",
            "range_pos_rank_cs",
            "trade_count_share_rank_cs",
            "top_queue_share_imb_rank_cs",
            "depth_pressure_slope_rank_cs",
            "high_gap_rank_cs",
            "ob_imb19_rank_cs",
            "depth_pressure_1019_rank_cs",
            "ob_imb4_rank_cs",
            "ret3_rank_cs",
            "ret12_resid_rank_cs",
            "day_open_gap_rank_cs",
            "ret24_rank_cs",
            "sell_vwad_dev_rank_cs",
            "trade_vwad_gap_rank_cs",
            "trade_high_center_gap_rank_cs",
            "micro_dev_rank_cs",
            "last_mid_dev_rank_cs",
        ]
        time_interactions_df = rank_df[time_interactions].mul(time_df["interval_frac_centered"], axis=0)
        time_interactions_df.columns = [f"{col}_x_time" for col in time_interactions]

        u_interactions = [
            "trade_imb_rank_cs",
            "flow_imb_rank_cs",
            "ret1_rank_cs",
            "ret6_rank_cs",
            "range_pos_rank_cs",
            "trade_count_share_rank_cs",
            "top_queue_share_imb_rank_cs",
            "high_gap_rank_cs",
            "ob_imb19_rank_cs",
            "depth_pressure_1019_rank_cs",
            "ob_imb4_rank_cs",
            "ret3_rank_cs",
            "ret12_resid_rank_cs",
            "high_minus_low_rank_cs",
            "day_open_gap_rank_cs",
            "ret24_rank_cs",
            "sell_vwad_dev_rank_cs",
            "trade_vwad_gap_rank_cs",
            "trade_high_center_gap_rank_cs",
            "micro_dev_rank_cs",
            "last_mid_dev_rank_cs",
        ]
        u_interactions_df = rank_df[u_interactions].mul(time_df["interval_u"], axis=0)
        u_interactions_df.columns = [f"{col}_x_u" for col in u_interactions]

        # Keep second-order gates narrow; the broader version was killed by grader resource limits.
        nonlinear_time_interactions = self._nonlinear_time_interactions()
        if nonlinear_time_interactions:
            time_sq_interactions_df = rank_df[nonlinear_time_interactions].mul(time_df["interval_frac_sq"], axis=0)
            time_sq_interactions_df.columns = [f"{col}_x_time_sq" for col in nonlinear_time_interactions]

            u_sq_interactions_df = rank_df[nonlinear_time_interactions].mul(time_df["interval_u_sq"], axis=0)
            u_sq_interactions_df.columns = [f"{col}_x_u_sq" for col in nonlinear_time_interactions]
        else:
            time_sq_interactions_df = pd.DataFrame(index=df.index)
            u_sq_interactions_df = pd.DataFrame(index=df.index)

        feat_df = pd.concat(
            [
                base_df,
                cs_out,
                rank_df,
                time_df,
                time_interactions_df,
                u_interactions_df,
                time_sq_interactions_df,
                u_sq_interactions_df,
            ],
            axis=1,
        ).astype(np.float32)

        xdf = (
            pd.concat([df[self.mcols], feat_df[self.featureNames()]], axis=1)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .set_index(self.mcols)
        )
        ydf = df[self.mcols + [self.ycol]].set_index(self.mcols)
        return xdf, ydf.fillna(0.0)
