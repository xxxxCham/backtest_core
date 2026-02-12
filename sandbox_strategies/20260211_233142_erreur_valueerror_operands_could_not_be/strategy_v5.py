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
        return {
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(
                name="RSI Overbought", type="int", min=50, max=90, default=70
            ),
            "rsi_oversold": ParameterSpec(
                name="RSI Oversold", type="int", min=10, max=50, default=30
            ),
            "rsi_period": ParameterSpec(
                name="RSI Period", type="int", min=7, max=28, default=14
            ),
            "stop_atr_mult": ParameterSpec(
                name="Stop ATR Multiplier",
                type="float",
                min=0.5,
                max=3.0,
                default=1.5,
            ),
            "tp_atr_mult": ParameterSpec(
                name="Take Profit ATR Multiplier",
                type="float",
                min=1.0,
                max=6.0,
                default=3.0,
            ),
            "warmup": ParameterSpec(
                name="Warm Up Period", type="int", min=20, max=100, default=50
            ),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        warmup = int(params.get("warmup", 50))
        
        # Extract indicators
        bb = indicators["bollinger"]
        upper_bollinger = np.nan_to_num(bb["upper"])
        lower_bollinger = np.nan_to_num(bb["lower"])
        rsi = np.nan_to_num(indicators["rsi"])
        
        # Close prices array
        close = df["close"].values
        
        # Entry conditions
        long_entry = (close > upper_bollinger) & (rsi < rsi_oversold)
        short_entry = (close < lower_bollinger) & (rsi > rsi_overbought)
        
        # Exit conditions based on current position
        exit_long = close < lower_bollinger
        exit_short = close > upper_bollinger
        
        # Apply warmup period
        signals.iloc[:warmup] = 0.0
        
        # Generate signals
        for i in range(warmup, len(signals)):
            if long_entry[i]:
                signals.iloc[i] = 1.0  # LONG entry
            elif short_entry[i]:
                signals.iloc[i] = -1.0  # SHORT entry
            else:
                if signals.iloc[i-1] == 1.0 and exit_long[i]:
                    signals.iloc[i] = 0.0  # Exit LONG
                elif signals.iloc[i-1] == -1.0 and exit_short[i]:
                    signals.iloc[i] = 0.0  # Exit SHORT
        
        return signals