from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bolinger_supertrend_atr_breakout_with_rsi_filter")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "supertrend", "atr", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(50, 100, 1, "Overbought RSI level"),
            "rsi_oversold": ParameterSpec(0, 50, 1, "Oversold RSI level"),
            "rsi_period": ParameterSpec(5, 30, 1, "RSI period"),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1, "Stop loss multiplier (ATR)"),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1, "Take profit multiplier (ATR)"),
            "warmup": ParameterSpec(20, 100, 1, "Warmup bars"),
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
        upper_band = np.nan_to_num(bb["upper"])
        lower_band = np.nan_to_num(bb["lower"])
        close = np.nan_to_num(df["close"].values)
        supertrend = np.nan_to_num(indicators["supertrend"]["supertrend"])
        supertrend_direction = np.nan_to_num(indicators["supertrend"]["direction"])
        atr = np.nan_to_num(indicators["atr"])
        rsi = np.nan_to_num(indicators["rsi"])
        
        # Parameters
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))
        
        # Entry conditions
        long_condition = (close > upper_band) & (supertrend_direction > 0) & (rsi < rsi_overbought)
        short_condition = (close < lower_band) & (supertrend_direction < 0) & (rsi > rsi_oversold)
        
        # Exit conditions
        exit_long_condition = (close < upper_band) & (supertrend_direction < 0)
        exit_short_condition = (close > lower_band) & (supertrend_direction > 0)
        
        # Initialize entry/exit arrays
        entry_long = np.zeros_like(close)
        entry_short = np.zeros_like(close)
        exit_long = np.zeros_like(close)
        exit_short = np.zeros_like(close)
        
        # Set entry signals
        entry_long[long_condition] = 1.0
        entry_short[short_condition] = -1.0
        
        # Set exit signals
        exit_long[exit_long_condition] = 1.0
        exit_short[exit_short_condition] = 1.0
        
        # Generate final signals
        positions = np.zeros_like(close)
        in_long = False
        in_short = False
        
        for i in range(len(close)):
            if entry_long[i] == 1.0:
                if not in_short:
                    positions[i] = 1.0
                    in_long = True
                    in_short = False
            elif entry_short[i] == -1.0:
                if not in_long:
                    positions[i] = -1.0
                    in_long = False
                    in_short = True
            elif exit_long[i] == 1.0 and in_long:
                positions[i] = 0.0
                in_long = False
            elif exit_short[i] == 1.0 and in_short:
                positions[i] = 0.0
                in_short = False
            elif in_long or in_short:
                positions[i] = positions[i-1] if i > 0 else 0.0
        
        signals = pd.Series(positions, index=df.index, dtype=np.float64)
        
        # Warmup protection
        signals.iloc[:warmup] = 0.0
        
        return signals