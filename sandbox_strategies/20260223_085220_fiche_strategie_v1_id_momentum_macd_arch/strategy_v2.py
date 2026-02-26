from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="momentum_macd")

    @property
    def required_indicators(self) -> List[str]:
        return ["macd", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "leverage": 1,
            "rsi_overbought": 75,
            "rsi_oversold": 40,
            "rsi_short_lower": 30,
            "rsi_short_upper": 60,
            "rsi_overbought_exit": 80,
            "rsi_oversold_exit": 20,
            "stop_atr_mult": 2.25,
            "tp_atr_mult": 5.5,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(
                name="rsi_period",
                min_val=5,
                max_val=50,
                default=20,
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
                default=5.5,
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

        # Wrap indicator arrays with np.nan_to_num
        macd = np.nan_to_num(indicators['macd']["macd"])
        signal_line = np.nan_to_num(indicators['macd']["signal"])
        hist = np.nan_to_num(indicators['macd']["histogram"])

        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])

        # Helper cross functions
        prev_macd = np.roll(macd, 1)
        prev_signal = np.roll(signal_line, 1)
        prev_macd[0] = np.nan
        prev_signal[0] = np.nan
        cross_up = (macd > signal_line) & (prev_macd <= prev_signal)
        cross_down = (macd < signal_line) & (prev_macd >= prev_signal)

        # Long entry conditions
        long_mask = (
            cross_up
            & (rsi > params["rsi_oversold"])
            & (rsi < params["rsi_overbought"])
        )

        # Short entry conditions
        short_mask = (
            cross_down
            & (rsi > params["rsi_short_lower"])
            & (rsi < params["rsi_short_upper"])
        )

        # Exit conditions
        prev_hist = np.roll(hist, 1)
        prev_hist[0] = np.nan
        hist_sign_change = (np.sign(hist) != np.sign(prev_hist)) & (~np.isnan(prev_hist))
        exit_mask = (
            hist_sign_change
            | (rsi > params["rsi_overbought_exit"])
            | (rsi < params["rsi_oversold_exit"])
        )

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # Warmup
        signals.iloc[:warmup] = 0.0

        # ATR based stop and take profit columns
        close = df["close"].values
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Long entry SL/TP
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - params["stop_atr_mult"] * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + params["tp_atr_mult"] * atr[long_mask]

        # Short entry SL/TP
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + params["stop_atr_mult"] * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - params["tp_atr_mult"] * atr[short_mask]

        return signals