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
        return {"adx_period": 14, "adx_smoothing_threshold": 20, "adx_threshold": 25, "atr_threshold": 100, "stop_atr_mult": 1.5, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(10, 30, 14),
            "adx_smoothing_threshold": ParameterSpec(10, 30, 20),
            "adx_threshold": ParameterSpec(10, 40, 25),
            "atr_threshold": ParameterSpec(50, 200, 100),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 1.5),
            "supertrend_multiplier": ParameterSpec(1.0, 5.0, 3.0),
            "supertrend_period": ParameterSpec(5, 20, 10),
            "tp_atr_mult": ParameterSpec(2.0, 6.0, 3.0),
            "warmup": ParameterSpec(20, 100, 50),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        supertrend_direction = np.nan_to_num(indicators["supertrend"]["direction"])
        adx_value = np.nan_to_num(indicators["adx"]["adx"])
        atr_value = np.nan_to_num(indicators["atr"])
        
        # Extract params
        adx_threshold = params.get("adx_threshold", 25)
        adx_smoothing_threshold = params.get("adx_smoothing_threshold", 20)
        atr_threshold = params.get("atr_threshold", 100)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        
        # Entry condition
        entry_condition = (
            (supertrend_direction > 0) &
            (adx_value > adx_threshold) &
            (adx_value > adx_smoothing_threshold) &
            (atr_value > atr_threshold)
        )
        
        # Exit condition
        exit_condition = (
            (supertrend_direction < 0) |
            (adx_value < adx_threshold)
        )
        
        # Generate signals
        entry_signals = np.where(entry_condition, 1.0, 0.0)
        exit_signals = np.where(exit_condition, -1.0, 0.0)
        
        # Combine signals
        signals = pd.Series(entry_signals, index=df.index, dtype=np.float64)
        signals = signals + pd.Series(exit_signals, index=df.index, dtype=np.float64)
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals