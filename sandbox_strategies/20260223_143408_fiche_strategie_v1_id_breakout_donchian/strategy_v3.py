from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="breakout_donchian_adx_rsi")

    @property
    def required_indicators(self) -> List[str]:
        return ["donchian", "adx", "atr", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "leverage": 1,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 2.5,
            "tp_atr_mult": 2.5,
            "warmup": 50,
            # thresholds used in the logic
            "adx_threshold": 25,
            "rsi_threshold": 50,
            "adx_exit_threshold": 20,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(
                name="rsi_period",
                min_val=5,
                max_val=50,
                default=14,
                param_type="int",
                step=1,
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                min_val=0.5,
                max_val=4.0,
                default=2.5,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=0.5,
                max_val=4.0,
                default=2.5,
                param_type="float",
                step=0.1,
            ),
            "leverage": ParameterSpec(
                name="leverage",
                min_val=1,
                max_val=2,
                default=1,
                param_type="int",
                step=1,
            ),
            # thresholds for entry/exit
            "adx_threshold": ParameterSpec(
                name="adx_threshold",
                min_val=10,
                max_val=50,
                default=25,
                param_type="int",
                step=1,
            ),
            "rsi_threshold": ParameterSpec(
                name="rsi_threshold",
                min_val=30,
                max_val=70,
                default=50,
                param_type="int",
                step=1,
            ),
            "adx_exit_threshold": ParameterSpec(
                name="adx_exit_threshold",
                min_val=10,
                max_val=30,
                default=20,
                param_type="int",
                step=1,
            ),
        }

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        n = len(df)
        warmup = int(params.get("warmup", 50))
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Extract indicator arrays
        close = df["close"].values
        dc = indicators['donchian']
        upper = np.nan_to_num(dc["upper"])
        middle = np.nan_to_num(dc["middle"])
        lower = np.nan_to_num(dc["lower"])

        adx_arr = indicators['adx']
        adx_val = np.nan_to_num(adx_arr["adx"])

        atr = np.nan_to_num(indicators['atr'])
        rsi = np.nan_to_num(indicators['rsi'])

        # Thresholds
        adx_thr = params.get("adx_threshold", 25)
        rsi_thr = params.get("rsi_threshold", 50)
        adx_exit_thr = params.get("adx_exit_threshold", 20)

        # Long / short entry masks
        long_cond = (
            (close > upper)
            & (adx_val > adx_thr)
            & (rsi > rsi_thr)
        )
        short_cond = (
            (close < lower)
            & (adx_val > adx_thr)
            & (rsi < rsi_thr)
        )

        signals[long_cond] = 1.0
        signals[short_cond] = -1.0

        # Exit condition (no explicit exit signal; SL/TP will be used)
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        cross_down = (close < middle) & (prev_close >= middle)
        exit_mask = cross_down | (adx_val < adx_exit_thr)
        # exit_mask is not used to modify signals directly

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Prepare SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # ATR-based SL/TP for long entries
        if long_cond.any():
            df.loc[long_cond, "bb_stop_long"] = (
                close[long_cond] - params["stop_atr_mult"] * atr[long_cond]
            )
            df.loc[long_cond, "bb_tp_long"] = (
                close[long_cond] + params["tp_atr_mult"] * atr[long_cond]
            )

        # ATR-based SL/TP for short entries
        if short_cond.any():
            df.loc[short_cond, "bb_stop_short"] = (
                close[short_cond] + params["stop_atr_mult"] * atr[short_cond]
            )
            df.loc[short_cond, "bb_tp_short"] = (
                close[short_cond] - params["tp_atr_mult"] * atr[short_cond]
            )

        return signals