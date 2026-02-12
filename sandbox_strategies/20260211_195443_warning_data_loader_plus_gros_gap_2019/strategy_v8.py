from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="rsi_bollinger_atr_filter")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"atr_threshold": 20, "rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "atr_threshold": ParameterSpec("atr_threshold", 10, 50, 20),
            "rsi_overbought": ParameterSpec("rsi_overbought", 60, 85, 70),
            "rsi_oversold": ParameterSpec("rsi_oversold", 15, 40, 30),
            "rsi_period": ParameterSpec("rsi_period", 5, 30, 14),
            "stop_atr_mult": ParameterSpec("stop_atr_mult", 1.0, 3.0, 1.5),
            "tp_atr_mult": ParameterSpec("tp_atr_mult", 2.0, 6.0, 3.0),
            "warmup": ParameterSpec("warmup", 20, 100, 50),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Extract indicators
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        upper_band = np.nan_to_num(bb["upper"])
        middle_band = np.nan_to_num(bb["middle"])
        lower_band = np.nan_to_num(bb["lower"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Parameters
        atr_threshold = params.get("atr_threshold", 20)
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        
        # Entry conditions
        close = np.nan_to_num(df["close"])
        long_condition = (close < lower_band) & (rsi < rsi_oversold) & (atr > atr_threshold)
        short_condition = (close > upper_band) & (rsi > rsi_overbought) & (atr > atr_threshold)
        
        # Exit conditions
        position = np.zeros_like(close)
        for i in range(1, len(df)):
            if position[i-1] != 0:
                if position[i-1] == 1 and close[i] > middle_band[i]:
                    position[i] = 0
                elif position[i-1] == -1 and close[i] < middle_band[i]:
                    position[i] = 0
                else:
                    position[i] = position[i-1]
            else:
                if long_condition[i]:
                    position[i] = 1
                elif short_condition[i]:
                    position[i] = -1
                else:
                    position[i] = 0
        
        # Generate signals
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        for i in range(1, len(df)):
            if position[i] == 1 and position[i-1] != 1:
                signals.iloc[i] = 1.0
            elif position[i] == -1 and position[i-1] != -1:
                signals.iloc[i] = -1.0
            elif position[i] == 0 and position[i-1] != 0:
                signals.iloc[i] = 0.0
                
        return signals