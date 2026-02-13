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
        return {"adx_period": 14, "adx_threshold": 25, "atr_threshold": 0.001, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(param_type="int", min_value=5, max_value=30, step=1),
            "adx_threshold": ParameterSpec(param_type="int", min_value=10, max_value=50, step=5),
            "atr_threshold": ParameterSpec(param_type="float", min_value=0.0001, max_value=0.01, step=0.0001),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=3.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=5.0, step=0.5),
            "warmup": ParameterSpec(param_type="int", min_value=20, max_value=100, step=10),
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
        
        # Get parameters
        adx_threshold = params.get("adx_threshold", 25)
        atr_threshold = params.get("atr_threshold", 0.001)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))
        
        # Entry condition
        entry_condition = (
            (supertrend_direction > 0) &
            (adx_value >= adx_threshold) &
            (atr_value > atr_threshold)
        )
        
        # Exit condition
        exit_condition = (
            (supertrend_direction < 0) |
            (adx_value < adx_threshold)
        )
        
        # Generate signals
        long_entries = entry_condition & ~np.roll(entry_condition, 1)
        long_exits = exit_condition & np.roll(entry_condition, 1)
        
        # Apply signals
        signals.loc[long_entries] = 1.0
        signals.loc[long_exits] = 0.0
        
        # Warmup protection
        signals.iloc[:warmup] = 0.0
        
        return signals