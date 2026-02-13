from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mean_reversion_bollinger_rsi_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(10, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 4.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        close = np.nan_to_num(df["close"].values)
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        bb_upper = np.nan_to_num(bb["upper"])
        bb_middle = np.nan_to_num(bb["middle"])
        bb_lower = np.nan_to_num(bb["lower"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Extract params
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))
        
        # Entry condition: price touches lower bollinger band with oversold rsi
        entry_condition = (close < bb_lower) & (rsi < rsi_oversold)
        
        # Additional filter: RSI should not be too oversold to avoid false signals
        entry_condition = entry_condition & (rsi > rsi_oversold + 10)
        
        # Exit condition: price returns to bollinger middle band
        exit_condition = close > bb_middle
        
        # Create signal array
        entry_signals = np.zeros_like(close)
        exit_signals = np.zeros_like(close)
        
        # Mark entry points
        entry_points = np.where(entry_condition)[0]
        for i in entry_points:
            if i > 0:
                entry_signals[i] = 1.0  # LONG signal
        
        # Mark exit points
        exit_points = np.where(exit_condition)[0]
        for i in exit_points:
            if i > 0:
                exit_signals[i] = -1.0  # Exit signal
        
        # Combine signals
        signals = pd.Series(entry_signals, index=df.index, dtype=np.float64)
        signals = signals + pd.Series(exit_signals, index=df.index, dtype=np.float64)
        
        # Set warmup period
        signals.iloc[:warmup] = 0.0
        
        # Convert to 1.0/-1.0 for long/exit only
        long_signals = signals.where(signals > 0, 0.0)
        exit_signals = signals.where(signals < 0, 0.0)
        
        # Ensure only one signal per bar
        final_signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        final_signals.iloc[:] = long_signals.iloc[:]
        final_signals = final_signals.where(~(exit_signals != 0), -1.0)
        
        return final_signals