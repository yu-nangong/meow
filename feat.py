import os
import numpy as np
import pandas as pd
from log import log


class MeowFeatureGenerator(object):
    @classmethod
    def featureNames(cls):
        return [
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
        ]

    def __init__(self, cacheDir):
        self.cacheDir = cacheDir
        self.ycol = "fret12"
        self.mcols = ["symbol", "date", "interval"]

    def genFeatures(self, df):
        log.inf("Generating {} features from raw data...".format(len(self.featureNames())))
        eps = 1e-6
        df = df.sort_values(self.mcols, kind="mergesort").copy()
        sym_day = df.groupby(["symbol", "date"], sort=False)

        df.loc[:, "ob_imb0"] = (df["bsize0"] - df["asize0"]) / (df["bsize0"] + df["asize0"] + eps)
        df.loc[:, "ob_imb4"] = (df["bsize0_4"] - df["asize0_4"]) / (df["bsize0_4"] + df["asize0_4"] + eps)
        df.loc[:, "ob_imb9"] = (df["bsize5_9"] - df["asize5_9"]) / (df["bsize5_9"] + df["asize5_9"] + eps)
        df.loc[:, "ob_imb19"] = (df["bsize10_19"] - df["asize10_19"]) / (df["bsize10_19"] + df["asize10_19"] + eps)
        df.loc[:, "ob_imb_front_back"] = df["ob_imb0"] - df["ob_imb19"]
        df.loc[:, "ob_imb_inner_outer"] = df["ob_imb4"] - 0.5 * (df["ob_imb9"] + df["ob_imb19"])
        df.loc[:, "trade_imb"] = (df["tradeBuyQty"] - df["tradeSellQty"]) / (df["tradeBuyQty"] + df["tradeSellQty"] + eps)
        df.loc[:, "turnover_imb"] = (
            (df["tradeBuyTurnover"] - df["tradeSellTurnover"])
            / (df["tradeBuyTurnover"] + df["tradeSellTurnover"] + eps)
        )
        df.loc[:, "add_imb"] = (df["addBuyQty"] - df["addSellQty"]) / (df["addBuyQty"] + df["addSellQty"] + eps)
        df.loc[:, "cxl_imb"] = (df["cxlBuyQty"] - df["cxlSellQty"]) / (df["cxlBuyQty"] + df["cxlSellQty"] + eps)
        df.loc[:, "add_turn_imb"] = (
            (df["addBuyTurnover"] - df["addSellTurnover"])
            / (df["addBuyTurnover"] + df["addSellTurnover"] + eps)
        )
        df.loc[:, "cxl_turn_imb"] = (
            (df["cxlBuyTurnover"] - df["cxlSellTurnover"])
            / (df["cxlBuyTurnover"] + df["cxlSellTurnover"] + eps)
        )
        df.loc[:, "flow_imb"] = (
            (df["addBuyQty"] - df["cxlBuyQty"]) - (df["addSellQty"] - df["cxlSellQty"])
        ) / (
            df["addBuyQty"] + df["cxlBuyQty"] + df["addSellQty"] + df["cxlSellQty"] + eps
        )
        df.loc[:, "trade_count_imb"] = (df["nTradeBuy"] - df["nTradeSell"]) / (df["nTradeBuy"] + df["nTradeSell"] + eps)
        df.loc[:, "add_count_imb"] = (df["nAddBuy"] - df["nAddSell"]) / (df["nAddBuy"] + df["nAddSell"] + eps)
        df.loc[:, "cxl_count_imb"] = (df["nCxlBuy"] - df["nCxlSell"]) / (df["nCxlBuy"] + df["nCxlSell"] + eps)
        df.loc[:, "depth_pressure_04"] = (df["btr0_4"] - df["atr0_4"]) / (df["btr0_4"] + df["atr0_4"] + eps)
        df.loc[:, "depth_pressure_59"] = (df["btr5_9"] - df["atr5_9"]) / (df["btr5_9"] + df["atr5_9"] + eps)
        df.loc[:, "depth_pressure_1019"] = (df["btr10_19"] - df["atr10_19"]) / (df["btr10_19"] + df["atr10_19"] + eps)
        df.loc[:, "depth_pressure_slope"] = df["depth_pressure_04"] - df["depth_pressure_1019"]
        df.loc[:, "depth_pressure_curve"] = df["depth_pressure_59"] - 0.5 * (
            df["depth_pressure_04"] + df["depth_pressure_1019"]
        )
        bid_depth_total = df["bsize0_4"] + df["bsize5_9"] + df["bsize10_19"]
        ask_depth_total = df["asize0_4"] + df["asize5_9"] + df["asize10_19"]
        df.loc[:, "bid_near_share"] = df["bsize0_4"] / (bid_depth_total + eps)
        df.loc[:, "ask_near_share"] = df["asize0_4"] / (ask_depth_total + eps)
        df.loc[:, "near_share_imb"] = df["bid_near_share"] - df["ask_near_share"]
        df.loc[:, "top_queue_share_imb"] = (
            df["bsize0"] / (bid_depth_total + eps)
        ) - (
            df["asize0"] / (ask_depth_total + eps)
        )
        df.loc[:, "spread"] = (df["ask0"] - df["bid0"]) / (df["midpx"] + eps)
        df.loc[:, "micro_dev"] = (
            (
                (df["ask0"] * df["bsize0"] + df["bid0"] * df["asize0"])
                / (df["asize0"] + df["bsize0"] + eps)
            )
            - df["midpx"]
        ) / (df["midpx"] + eps)
        df.loc[:, "last_mid_dev"] = (df["lastpx"] - df["midpx"]) / (df["midpx"] + eps)
        df.loc[:, "buy_vwad_dev"] = (df["buyVwad"] - df["midpx"]) / (df["midpx"] + eps)
        df.loc[:, "sell_vwad_dev"] = (df["sellVwad"] - df["midpx"]) / (df["midpx"] + eps)
        df.loc[:, "trade_vwad_gap"] = (df["buyVwad"] - df["sellVwad"]) / (df["midpx"] + eps)
        df.loc[:, "high_gap"] = (df["high"] - df["midpx"]) / (df["midpx"] + eps)
        df.loc[:, "low_gap"] = (df["midpx"] - df["low"]) / (df["midpx"] + eps)
        df.loc[:, "high_minus_low"] = (df["high"] - df["low"]) / (df["midpx"] + eps)
        df.loc[:, "trade_buy_high_gap"] = (df["tradeBuyHigh"] - df["midpx"]) / (df["midpx"] + eps)
        df.loc[:, "trade_sell_high_gap"] = (df["tradeSellHigh"] - df["midpx"]) / (df["midpx"] + eps)
        df.loc[:, "vwad_center_dev"] = 0.5 * (df["buy_vwad_dev"] + df["sell_vwad_dev"])
        df.loc[:, "trade_high_center_gap"] = 0.5 * (df["trade_buy_high_gap"] + df["trade_sell_high_gap"])
        df.loc[:, "trade_high_skew"] = df["trade_buy_high_gap"] - df["trade_sell_high_gap"]
        df.loc[:, "high_vs_trade_high_gap"] = df["high_gap"] - df["trade_high_center_gap"]
        df.loc[:, "day_open_gap"] = (df["midpx"] - df["open"]) / (df["open"] + eps)
        df.loc[:, "range_pos"] = (
            (df["midpx"] - df["low"]) - (df["high"] - df["midpx"])
        ) / (df["high"] - df["low"] + eps)
        df.loc[:, "trade_count_share"] = (
            df["nTradeBuy"] + df["nTradeSell"]
        ) / (
            df["nAddBuy"] + df["nAddSell"] + df["nCxlBuy"] + df["nCxlSell"] + df["nTradeBuy"] + df["nTradeSell"] + eps
        )

        df.loc[:, "ret1"] = sym_day["midpx"].pct_change(1)
        df.loc[:, "ret3"] = sym_day["midpx"].pct_change(3)
        df.loc[:, "ret6"] = sym_day["midpx"].pct_change(6)
        df.loc[:, "ret12"] = sym_day["midpx"].pct_change(12)
        df.loc[:, "ret24"] = sym_day["midpx"].pct_change(24)

        df.loc[:, "trade_imb_ema6"] = sym_day["trade_imb"].transform(
            lambda s: s.ewm(halflife=6, adjust=False).mean()
        )
        df.loc[:, "ob_imb0_ema6"] = sym_day["ob_imb0"].transform(
            lambda s: s.ewm(halflife=6, adjust=False).mean()
        )
        df.loc[:, "micro_dev_ema6"] = sym_day["micro_dev"].transform(
            lambda s: s.ewm(halflife=6, adjust=False).mean()
        )

        df.loc[:, "ret12_resid"] = df["ret12"] - df.groupby(["date", "interval"], sort=False)["ret12"].transform("mean")
        df.loc[:, "ret3_x_flow"] = df["ret3"] * df["flow_imb"]
        df.loc[:, "ret6_x_flow"] = df["ret6"] * df["flow_imb"]

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
        cs_means = df.groupby(["date", "interval"], sort=False)[cs_cols].transform("mean")
        for col in cs_cols:
            df.loc[:, f"{col}_cs"] = df[col] - cs_means[col]

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
        rank_df = df.groupby(["date", "interval"], sort=False)[rank_cols].rank(pct=True)
        for col in rank_cols:
            df.loc[:, f"{col}_rank_cs"] = rank_df[col] - 0.5

        xdf = (
            df[self.mcols + self.featureNames()]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .set_index(self.mcols)
        )
        ydf = df[self.mcols + [self.ycol]].set_index(self.mcols)
        return xdf, ydf.fillna(0.0)
