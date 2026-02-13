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
            "rsi_overbought": ParameterSpec(50, 90, 1),
            "rsi_oversold": ParameterSpec(10, 50, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
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
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_middle = np.nan_to_num(indicators["bollinger"]["middle"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Parameters
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        warmup = int(params.get("warmup", 50))
        
        # Entry condition: price touches upper Bollinger band and RSI is overbought
        entry_condition = (close >= bb_upper * 0.99) & (rsi > rsi_overbought)
        
        # Additional confirmation: RSI is decreasing
        rsi_condition = rsi < np.roll(rsi, 1)
        entry_condition = entry_condition & rsi_condition
        
        # Exit condition: price returns to middle Bollinger band
        exit_condition = close <= bb_middle
        
        # Generate signals
        entry_signals = np.where(entry_condition, 1.0, 0.0)
        exit_signals = np.where(exit_condition, -1.0, 0.0)
        
        # Combine signals
        signals = pd.Series(entry_signals, index=df.index, dtype=np.float64)
        signals = signals + pd.Series(exit_signals, index=df.index, dtype=np.float64)
        
        # Set warmup period
        signals.iloc[:warmup] = 0.0
        
        return signals