from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mean_reversion_bollinger_stoch_rsi_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "stoch_rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(70, 80, 1, "RSI Overbought Level"),
            "rsi_oversold": ParameterSpec(20, 30, 1, "RSI Oversold Level"),
            "rsi_period": ParameterSpec(10, 14, 1, "RSI Period"),
            "stop_atr_mult": ParameterSpec(1.0, 1.5, 0.1, "Stop Loss ATR Multiplier"),
            "tp_atr_mult": ParameterSpec(2.0, 3.0, 0.1, "Take Profit ATR Multiplier"),
            "warmup": ParameterSpec(30, 50, 1, "Warmup Bars"),
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
        stoch_rsi = indicators["stoch_rsi"]
        atr = indicators["atr"]
        
        # Extract components from dict indicators
        close = np.nan_to_num(df["close"].values)
        bb_upper = np.nan_to_num(bb["upper"])
        bb_middle = np.nan_to_num(bb["middle"])
        bb_lower = np.nan_to_num(bb["lower"])
        stoch_rsi_k = np.nan_to_num(stoch_rsi["k"])
        stoch_rsi_d = np.nan_to_num(stoch_rsi["d"])
        atr_values = np.nan_to_num(atr)
        
        # Entry conditions
        entry_long = (close < bb_lower) & (stoch_rsi_k < 20) & (close < bb_middle)
        
        # Exit condition
        exit_long = close > bb_middle
        
        # Find entry points
        entry_points = np.where(entry_long)[0]
        exit_points = np.where(exit_long)[0]
        
        # Initialize signal array
        signal_array = np.zeros(len(df))
        
        # Track active positions
        in_position = False
        entry_index = -1
        
        for i in range(len(df)):
            if not in_position and entry_long[i]:
                # Enter long position
                in_position = True
                entry_index = i
                signal_array[i] = 1.0
            elif in_position and exit_long[i]:
                # Exit long position
                in_position = False
                signal_array[i] = 0.0
            elif in_position:
                # Check for stop loss or take profit
                if i > entry_index:
                    # Calculate ATR-based stop loss and take profit
                    stop_loss = close[entry_index] - (atr_values[entry_index] * params["stop_atr_mult"])
                    take_profit = close[entry_index] + (atr_values[entry_index] * params["tp_atr_mult"])
                    
                    # Exit if stop loss or take profit is hit
                    if close[i] <= stop_loss:
                        in_position = False
                        signal_array[i] = 0.0
                    elif close[i] >= take_profit:
                        in_position = False
                        signal_array[i] = 0.0
        
        # Set signals
        signals = pd.Series(signal_array, index=df.index, dtype=np.float64)
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals