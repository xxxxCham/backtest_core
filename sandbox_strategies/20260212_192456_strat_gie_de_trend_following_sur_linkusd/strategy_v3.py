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
            "adx_threshold": ParameterSpec(10, 40, 1),
            "stop_atr_mult": ParameterSpec(0.5, 2.0, 0.1),
            "supertrend_multiplier": ParameterSpec(1.0, 5.0, 0.1),
            "supertrend_period": ParameterSpec(5, 20, 1),
            "tp_atr_mult": ParameterSpec(1.0, 4.0, 0.1),
            "warmup": ParameterSpec(20, 100, 5),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        supertrend = indicators["supertrend"]
        adx = indicators["adx"]
        atr = indicators["atr"]
        
        # Extract supertrend values
        supertrend_line = np.nan_to_num(supertrend["supertrend"])
        supertrend_direction = np.nan_to_num(supertrend["direction"])
        
        # Extract ADX values
        adx_value = np.nan_to_num(adx["adx"])
        
        # Extract ATR
        atr_values = np.nan_to_num(atr)
        
        # Get parameters
        adx_threshold = params.get("adx_threshold", 25)
        stop_atr_mult = params.get("stop_atr_mult", 1.0)
        tp_atr_mult = params.get("tp_atr_mult", 2.0)
        
        # Entry condition: supertrend direction up and ADX above threshold
        entry_condition = (supertrend_line < df["close"].values) & (supertrend_direction > 0) & (adx_value > adx_threshold)
        
        # Exit condition: supertrend direction down or ADX below threshold
        exit_condition = (supertrend_line > df["close"].values) | (adx_value < 20)
        
        # Initialize entry and exit signals
        entry_signal = pd.Series(0.0, index=df.index, dtype=np.float64)
        exit_signal = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Set entry signals
        entry_signal[entry_condition] = 1.0
        
        # Set exit signals
        exit_signal[exit_condition] = -1.0
        
        # Combine signals
        signals = entry_signal + exit_signal
        
        # Set warmup period
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals