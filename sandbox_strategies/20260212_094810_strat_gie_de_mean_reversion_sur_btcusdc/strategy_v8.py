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
            "rsi_overbought": ParameterSpec(60, 90, 1, "RSI overbought level"),
            "rsi_oversold": ParameterSpec(10, 40, 1, "RSI oversold level"),
            "rsi_period": ParameterSpec(5, 30, 1, "RSI period"),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1, "Stop-loss multiplier in ATR"),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1, "Take-profit multiplier in ATR"),
            "warmup": ParameterSpec(20, 100, 1, "Warmup period"),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))
        
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        bb_upper = np.nan_to_num(bb["upper"])
        bb_middle = np.nan_to_num(bb["middle"])
        bb_lower = np.nan_to_num(bb["lower"])
        atr = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)
        
        # Entry condition: price touches lower band and RSI is oversold
        entry_long = (close < bb_lower) & (rsi < rsi_oversold) & (close > bb_middle)
        
        # Exit condition: price crosses back to middle band
        exit_long = close > bb_middle
        
        # Generate signals
        entry_indices = np.where(entry_long)[0]
        exit_indices = np.where(exit_long)[0]
        
        # Set signals for entries
        for i in entry_indices:
            if i > 0:
                signals.iloc[i] = 1.0  # LONG signal
                
        # Set signals for exits
        for i in exit_indices:
            if i > 0 and signals.iloc[i-1] == 1.0:
                signals.iloc[i] = 0.0  # FLAT signal
                
        # Apply warmup
        signals.iloc[:warmup] = 0.0
        return signals