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
            "stop_atr_mult": ParameterSpec(1.0, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 4.0, 0.1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        adx_threshold = params.get("adx_threshold", 20)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        supertrend_direction = np.nan_to_num(indicators["supertrend"]["direction"])
        adx_value = np.nan_to_num(indicators["adx"]["adx"])
        atr_value = np.nan_to_num(indicators["atr"])
        
        # Entry condition: supertrend direction up and ADX above threshold
        entry_condition = (supertrend_direction > 0) & (adx_value > adx_threshold)
        
        # Exit condition: supertrend direction down
        exit_condition = supertrend_direction < 0
        
        # Generate long signals
        entry_points = np.where(entry_condition, 1.0, 0.0)
        exit_points = np.where(exit_condition, -1.0, 0.0)
        
        # Combine entry and exit
        signals = pd.Series(entry_points, index=df.index, dtype=np.float64)
        signals = signals.where(~exit_condition, -1.0)
        
        # Set warmup period to flat
        signals.iloc[:warmup] = 0.0
        
        return signals