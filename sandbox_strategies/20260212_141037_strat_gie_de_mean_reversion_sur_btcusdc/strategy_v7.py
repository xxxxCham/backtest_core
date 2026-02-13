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
            "warmup": ParameterSpec(10, 100, 1)
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
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        warmup = int(params.get("warmup", 50))
        
        rsi = np.nan_to_num(indicators["rsi"])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_middle = np.nan_to_num(indicators["bollinger"]["middle"])
        atr = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)
        
        # Entry condition: price touches upper bollinger band with overbought rsi and rsi is decreasing
        entry_condition = (close >= bb_upper * 0.99) & (rsi > rsi_overbought) & (rsi < np.roll(rsi, 1)) & (atr > np.roll(atr, 1))
        
        # Exit condition: price returns to bollinger middle band
        exit_condition = close <= bb_middle
        
        # Generate long signals
        long_entries = np.where(entry_condition, 1.0, 0.0)
        long_exits = np.where(exit_condition, -1.0, 0.0)
        
        # Combine entries and exits
        signals = pd.Series(long_entries, index=df.index, dtype=np.float64)
        signals = signals + pd.Series(long_exits, index=df.index, dtype=np.float64)
        
        # Set warmup period to flat signals
        signals.iloc[:warmup] = 0.0
        
        return signals