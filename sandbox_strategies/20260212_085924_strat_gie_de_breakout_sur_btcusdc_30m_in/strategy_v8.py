from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bolinger_supertrend_atr_breakout_revised")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "supertrend", "atr", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(50, 100, 1, "RSI Overbought Level"),
            "rsi_oversold": ParameterSpec(0, 50, 1, "RSI Oversold Level"),
            "rsi_period": ParameterSpec(5, 30, 1, "RSI Period"),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.1, "Stop Loss Multiplier (ATR)"),
            "tp_atr_mult": ParameterSpec(2.0, 10.0, 0.1, "Take Profit Multiplier (ATR)"),
            "warmup": ParameterSpec(10, 100, 1, "Warmup Period"),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        bb = indicators["bollinger"]
        st = indicators["supertrend"]
        atr = np.nan_to_num(indicators["atr"])
        rsi = np.nan_to_num(indicators["rsi"])
        
        # Extract Bollinger bands
        bb_upper = np.nan_to_num(bb["upper"])
        bb_lower = np.nan_to_num(bb["lower"])
        bb_middle = np.nan_to_num(bb["middle"])
        
        # Extract Supertrend
        st_direction = np.nan_to_num(st["direction"])
        
        # Previous values
        prev_close = np.roll(df["close"].values, 1)
        prev_bb_upper = np.roll(bb_upper, 1)
        prev_bb_lower = np.roll(bb_lower, 1)
        prev_st_direction = np.roll(st_direction, 1)
        prev_rsi = np.roll(rsi, 1)
        
        # Entry conditions
        entry_condition = (
            (df["close"].values > bb_upper) &
            (prev_close <= prev_bb_upper) &
            (st_direction == 1) &
            (rsi < params["rsi_overbought"])
        )
        
        # Exit conditions
        exit_condition = (
            (df["close"].values < bb_lower) |
            (st_direction == -1) |
            (rsi > params["rsi_overbought"])
        )
        
        # Set signals
        signals[entry_condition] = 1.0
        signals[exit_condition] = 0.0
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals