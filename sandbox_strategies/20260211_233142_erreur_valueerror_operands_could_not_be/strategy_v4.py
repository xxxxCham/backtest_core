from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="trend_rsi_bollinger")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(
                name="RSI Overbought",
                min=50,
                max=80,
                default=70,
                type=float
            ),
            "rsi_oversold": ParameterSpec(
                name="RSI Oversold",
                min=20,
                max=50,
                default=30,
                type=float
            ),
            "rsi_period": ParameterSpec(
                name="RSI Period",
                min=10,
                max=20,
                default=14,
                type=int
            ),
            "stop_atr_mult": ParameterSpec(
                name="Stop ATR Multiplier",
                min=1.0,
                max=3.0,
                default=1.5,
                type=float
            ),
            "tp_atr_mult": ParameterSpec(
                name="Take Profit ATR Multiplier",
                min=2.0,
                max=4.0,
                default=3.0,
                type=float
            ),
            "warmup": ParameterSpec(
                name="Warmup Period",
                min=30,
                max=60,
                default=50,
                type=int
            )
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        if len(df) <= warmup:
            return signals
        
        # Extract indicators with proper handling
        rsi = np.nan_to_num(indicators["rsi"])
        bollinger = indicators["bollinger"]
        upper_bollinger = np.nan_to_num(bollinger["upper"])
        lower_bollinger = np.nan_to_num(bollinger["lower"])
        
        # Ensure all arrays are the same length
        min_length = min(len(rsi), len(upper_bollinger), len(lower_bollinger))
        rsi = rsi[:min_length]
        upper_bollinger = upper_bollinger[:min_length]
        lower_bollinger = lower_bollinger[:min_length]
        
        # Entry conditions
        long_entry = (df['close'].values[:min_length] > upper_bollinger) & (rsi < params["rsi_oversold"])
        short_entry = (df['close'].values[:min_length] < lower_bollinger) & (rsi > params["rsi_overbought"])
        
        # Exit conditions
        exit_long = df['close'].values[:min_length] < lower_bollinger
        exit_short = df['close'].values[:min_length] > upper_bollinger
        
        # Apply signals
        signals.iloc[:min_length] = 0.0  # Neutral by default
        signals.iloc[:min_length][long_entry] = 1.0  # LONG signal
        signals.iloc[:min_length][short_entry] = -1.0  # SHORT signal
        
        # Exit existing positions
        current_position = 0.0
        for i in range(len(signals)):
            if signals[i] == 1.0 or signals[i] == -1.0:
                current_position = signals[i]
            elif exit_long[i] and current_position == 1.0:
                current_position = 0.0
                signals[i] = 0.0
            elif exit_short[i] and current_position == -1.0:
                current_position = 0.0
                signals[i] = 0.0
        
        return signals