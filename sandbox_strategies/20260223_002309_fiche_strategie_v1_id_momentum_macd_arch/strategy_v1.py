from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="momentum_macd_adx")

    @property
    def required_indicators(self) -> List[str]:
        return ["macd", "rsi", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "leverage": 1,
            "rsi_overbought": 55,
            "rsi_oversold": 45,
            "rsi_period": 18,
            "stop_atr_mult": 2.25,
            "tp_atr_mult": 5.0,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(
                name="rsi_period",
                min_val=5,
                max_val=50,
                default=18,
                param_type="int",
                step=1,
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                min_val=0.5,
                max_val=4.0,
                default=2.25,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=1.0,
                max_val=10.0,
                default=5.0,
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
        }

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get("warmup", 50))

        # unpack indicators
        macd_dict = indicators['macd']
        macd = np.nan_to_num(macd_dict["macd"])
        signal = np.nan_to_num(macd_dict["signal"])
        macd_hist = np.nan_to_num(macd_dict["histogram"])

        rsi = np.nan_to_num(indicators['rsi'])
        adx = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # helper for cross detection
        def cross_up(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x > y) & (prev_x <= prev_y)

        def cross_down(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x < y) & (prev_x >= prev_y)

        # Parameters for RSI and ADX thresholds with fallbacks
        rsi_long_low = params.get("rsi_long_low", 45)
        rsi_long_high = params.get("rsi_long_high", 55)
        rsi_short_low = params.get("rsi_short_low", 30)
        rsi_short_high = params.get("rsi_short_high", 55)
        adx_threshold = params.get("adx_threshold", 25)
        adx_exit_threshold = params.get("adx_exit_threshold", 20)

        # entry conditions
        long_mask = (
            cross_up(macd, signal)
            & (rsi > rsi_long_low)
            & (rsi < rsi_long_high)
            & (adx > adx_threshold)
        )
        short_mask = (
            cross_down(macd, signal)
            & (rsi > rsi_short_low)
            & (rsi < rsi_short_high)
            & (adx > adx_threshold)
        )

        # exit conditions
        hist_sign = np.sign(macd_hist)
        prev_hist_sign = np.roll(hist_sign, 1)
        prev_hist_sign[0] = np.nan
        sign_change = (hist_sign != prev_hist_sign) & ~np.isnan(prev_hist_sign)
        exit_mask = (
            sign_change
            | (rsi > 80.0)
            | (rsi < 20.0)
            | (adx < adx_exit_threshold)
        )

        # apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based stop‑loss and take‑profit
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - params["stop_atr_mult"] * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + params["tp_atr_mult"] * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + params["stop_atr_mult"] * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - params["tp_atr_mult"] * atr[short_mask]

        return signals