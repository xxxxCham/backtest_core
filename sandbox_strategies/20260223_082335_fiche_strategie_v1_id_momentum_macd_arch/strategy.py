from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="momentum_macd_adx_filter")

    @property
    def required_indicators(self) -> List[str]:
        return ["macd", "rsi", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "leverage": 1,
            "rsi_overbought": 65,
            "rsi_oversold": 35,
            "rsi_exit_overbought": 80,
            "rsi_exit_oversold": 20,
            "stop_atr_mult": 2.0,
            "tp_atr_mult": 6.0,
            "warmup": 50,
            "adx_threshold": 25,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(
                name="rsi_period",
                min_val=5,
                max_val=50,
                default=21,
                param_type="int",
                step=1,
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                min_val=0.5,
                max_val=4.0,
                default=2.0,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=1.0,
                max_val=10.0,
                default=6.0,
                param_type="float",
                step=0.1,
            ),
            "adx_period": ParameterSpec(
                name="adx_period",
                min_val=5,
                max_val=30,
                default=14,
                param_type="int",
                step=1,
            ),
            "leverage": ParameterSpec(
                name="leverage",
                min_val=1,
                max_val=2,
                default=1,
                param_type="int",
                step=1,
            ),
        }

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get("warmup", 50))

        # Unwrap indicators
        macd_dict = indicators['macd']
        macd = np.nan_to_num(macd_dict["macd"])
        signal_line = np.nan_to_num(macd_dict["signal"])
        hist = np.nan_to_num(macd_dict["histogram"])

        rsi = np.nan_to_num(indicators['rsi'])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])

        close = df["close"].values

        # Cross helpers
        prev_macd = np.roll(macd, 1)
        prev_signal = np.roll(signal_line, 1)
        prev_macd[0] = np.nan
        prev_signal[0] = np.nan
        cross_up = (macd > signal_line) & (prev_macd <= prev_signal)
        cross_down = (macd < signal_line) & (prev_macd >= prev_signal)

        # Entry conditions
        long_mask = (
            cross_up
            & (rsi > params["rsi_oversold"])
            & (rsi < params["rsi_overbought"])
            & (adx_val > params.get("adx_threshold", 25))
        )
        short_mask = (
            cross_down
            & (rsi > params["rsi_oversold"])
            & (rsi < params["rsi_overbought"])
            & (adx_val > params.get("adx_threshold", 25))
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        hist_sign = np.sign(hist)
        prev_hist_sign = np.roll(hist_sign, 1)
        prev_hist_sign[0] = 0
        sign_change = (hist_sign != prev_hist_sign) & (prev_hist_sign != 0)
        exit_mask = (
            sign_change
            | (rsi > params.get("rsi_exit_overbought", 80))
            | (rsi < params.get("rsi_exit_oversold", 20))
        )
        # signals[exit_mask] = 0.0  # optional: explicit flat

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        if long_mask.any():
            df.loc[long_mask, "bb_stop_long"] = close[long_mask] - params["stop_atr_mult"] * atr[long_mask]
            df.loc[long_mask, "bb_tp_long"] = close[long_mask] + params["tp_atr_mult"] * atr[long_mask]
        if short_mask.any():
            df.loc[short_mask, "bb_stop_short"] = close[short_mask] + params["stop_atr_mult"] * atr[short_mask]
            df.loc[short_mask, "bb_tp_short"] = close[short_mask] - params["tp_atr_mult"] * atr[short_mask]

        signals.iloc[:warmup] = 0.0
        return signals