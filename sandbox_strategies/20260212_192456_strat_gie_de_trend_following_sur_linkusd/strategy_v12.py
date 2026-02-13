from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="supertrend_adx_trend_following")

    @property
    def required_indicators(self) -> List[str]:
        return ["supertrend", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"adx_period": 14, "adx_threshold": 25.0, "stop_atr_mult": 1.0, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec("adx_period", 5, 30, 14),
            "adx_threshold": ParameterSpec("adx_threshold", 10.0, 40.0, 25.0),
            "stop_atr_mult": ParameterSpec("stop_atr_mult", 0.5, 3.0, 1.0),
            "supertrend_multiplier": ParameterSpec("supertrend_multiplier", 1.0, 5.0, 3.0),
            "supertrend_period": ParameterSpec("supertrend_period", 5, 30, 10),
            "tp_atr_mult": ParameterSpec("tp_atr_mult", 1.0, 5.0, 2.0),
            "warmup": ParameterSpec("warmup", 20, 100, 50),
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
        
        # Entry conditions
        adx_threshold = params.get("adx_threshold", 25.0)
        entry_condition = (supertrend_direction > 0) & (adx_value >= adx_threshold)
        
        # Exit conditions
        exit_condition = (supertrend_direction < 0) | (adx_value < 20.0)
        
        # Generate signals
        signals[entry_condition] = 1.0
        signals[exit_condition] = 0.0
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals