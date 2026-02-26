from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="lineausdc_30m_momentum_atr_new")

    @property
    def required_indicators(self) -> List[str]:
        return ["macd", "momentum", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "atr_period": 14,
            "leverage": 1,
            "macd_fast_period": 12,
            "macd_signal_period": 9,
            "macd_slow_period": 26,
            "momentum_period": 14,
            "stop_atr_mult": 2.2,
            "tp_atr_mult": 2.8,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "macd_fast_period": ParameterSpec(
                name="macd_fast_period",
                min_val=5,
                max_val=20,
                default=12,
                param_type="int",
                step=1,
            ),
            "macd_slow_period": ParameterSpec(
                name="macd_slow_period",
                min_val=15,
                max_val=50,
                default=26,
                param_type="int",
                step=1,
            ),
            "macd_signal_period": ParameterSpec(
                name="macd_signal_period",
                min_val=3,
                max_val=15,
                default=9,
                param_type="int",
                step=1,
            ),
            "momentum_period": ParameterSpec(
                name="momentum_period",
                min_val=5,
                max_val=30,
                default=14,
                param_type="int",
                step=1,
            ),
            "atr_period": ParameterSpec(
                name="atr_period",
                min_val=5,
                max_val=30,
                default=14,
                param_type="int",
                step=1,
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                min_val=0.5,
                max_val=4.0,
                default=2.2,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=1.0,
                max_val=5.0,
                default=2.8,
                param_type="float",
                step=0.1,
            ),
            "warmup": ParameterSpec(
                name="warmup",
                min_val=20,
                max_val=100,
                default=50,
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
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        n = len(df)
        warmup = int(params.get("warmup", 50))
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Convert indicator arrays to numeric arrays
        macd_arr = np.nan_to_num(indicators['macd']["macd"])
        signal_line = np.nan_to_num(indicators['macd']["signal"])
        hist = np.nan_to_num(indicators['macd']["histogram"])
        momentum = np.nan_to_num(indicators['momentum'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Helper functions for cross detection
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

        # Entry and exit conditions
        long_entry = (macd_arr > signal_line) & (hist > 0) & (momentum > 0)
        short_entry = (macd_arr < signal_line) & (hist < 0) & (momentum < 0)

        # For histogram crossing zero use a zero array of same length
        zero_arr = np.zeros_like(hist)
        long_exit = cross_down(macd_arr, signal_line) | cross_down(hist, zero_arr)
        short_exit = cross_up(macd_arr, signal_line) | cross_up(hist, zero_arr)

        # Build masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        long_mask[long_entry] = True
        short_mask[short_entry] = True
        exit_mask = long_exit | short_exit

        # Apply signals
        signals[exit_mask] = 0.0
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals.iloc[:warmup] = 0.0

        # Stop/TP columns
        stop_atr_mult = params.get("stop_atr_mult", 2.2)
        tp_atr_mult = params.get("tp_atr_mult", 2.8)

        df["bb_stop_long"] = np.nan
        df["bb_tp_long"] = np.nan
        df["bb_stop_short"] = np.nan
        df["bb_tp_short"] = np.nan

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]

        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        return signals