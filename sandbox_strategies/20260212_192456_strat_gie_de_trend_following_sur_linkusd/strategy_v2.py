from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="supertrend_adx_atr_trend_following")

    @property
    def required_indicators(self) -> List[str]:
        return ["supertrend", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"adx_period": 14, "adx_threshold": 25, "stop_atr_mult": 1.0, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(10, 30, 1),
            "adx_threshold": ParameterSpec(10, 50, 1),
            "stop_atr_mult": ParameterSpec(0.5, 3.0, 0.1),
            "supertrend_multiplier": ParameterSpec(1.0, 5.0, 0.1),
            "supertrend_period": ParameterSpec(5, 20, 1),
            "tp_atr_mult": ParameterSpec(1.0, 5.0, 0.1),
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
        supertrend_data = indicators["supertrend"]
        adx_data = indicators["adx"]
        atr_values = np.nan_to_num(indicators["atr"])
        
        # Extract components
        supertrend = np.nan_to_num(supertrend_data["supertrend"])
        direction = np.nan_to_num(supertrend_data["direction"])
        adx = np.nan_to_num(adx_data["adx"])
        
        # Get params
        adx_threshold = params.get("adx_threshold", 25)
        warmup = int(params.get("warmup", 50))
        
        # Entry condition: supertrend direction up, ADX above threshold
        entry_condition = (supertrend < df["close"].values) & (direction > 0) & (adx > adx_threshold)
        
        # Exit condition: supertrend direction down or ADX below 20
        exit_condition = (supertrend > df["close"].values) | (adx < 20)
        
        # Generate signals
        long_entries = entry_condition
        long_exits = exit_condition
        
        # Create signal series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        signals.iloc[:warmup] = 0.0
        
        # Apply long entries and exits
        entry_indices = np.where(long_entries)[0]
        exit_indices = np.where(long_exits)[0]
        
        # Mark long signals
        for i in entry_indices:
            if i < len(signals):
                signals.iloc[i] = 1.0
                
        # Mark exits (flat)
        for i in exit_indices:
            if i < len(signals):
                signals.iloc[i] = 0.0
                
        return signals