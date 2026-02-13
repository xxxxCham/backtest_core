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
        return {"adx_threshold": 20, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_threshold": ParameterSpec(10, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
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
        supertrend_direction = np.nan_to_num(indicators["supertrend"]["direction"])
        adx_value = np.nan_to_num(indicators["adx"]["adx"])
        adx_plus_di = np.nan_to_num(indicators["adx"]["plus_di"])
        adx_minus_di = np.nan_to_num(indicators["adx"]["minus_di"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Extract params
        adx_threshold = params.get("adx_threshold", 20)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))
        
        # Warmup protection
        signals.iloc[:warmup] = 0.0
        
        # Entry conditions: long only
        entry_condition = (
            (supertrend_direction > 0) &
            (adx_value > adx_threshold) &
            (adx_plus_di > adx_minus_di)
        )
        
        # Exit conditions
        exit_condition = (
            (supertrend_direction < 0) |
            (adx_value < adx_threshold) |
            (adx_plus_di < adx_minus_di)
        )
        
        # Generate signals
        long_entries = entry_condition & ~np.roll(entry_condition, 1)
        long_exits = exit_condition & np.roll(entry_condition, 1)
        
        # Apply signals
        signals[long_entries] = 1.0
        signals[long_exits] = 0.0
        
        return signals