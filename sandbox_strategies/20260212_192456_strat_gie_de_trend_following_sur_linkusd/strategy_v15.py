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
        return {"adx_period": 14, "adx_threshold": 25.0, "atr_threshold": 0.001, "stop_atr_mult": 1.0, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(10, 30, 1),
            "adx_threshold": ParameterSpec(10.0, 40.0, 1.0),
            "atr_threshold": ParameterSpec(0.0001, 0.01, 0.0001),
            "stop_atr_mult": ParameterSpec(0.5, 2.0, 0.5),
            "tp_atr_mult": ParameterSpec(1.0, 4.0, 0.5),
            "warmup": ParameterSpec(20, 100, 10)
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
        
        # Apply filters
        adx_threshold = params["adx_threshold"]
        atr_threshold = params["atr_threshold"]
        
        # Entry condition: supertrend direction up, adx above threshold, atr above threshold
        entry_condition = (supertrend_direction > 0) & (adx_value >= adx_threshold) & (atr_value > atr_threshold)
        
        # Exit condition: supertrend direction down
        exit_condition = supertrend_direction < 0
        
        # Generate signals
        position = 0
        for i in range(len(df)):
            if entry_condition[i] and position == 0:
                position = 1  # LONG
            elif exit_condition[i] and position == 1:
                position = 0  # FLAT
            
            if position == 1:
                signals.iloc[i] = 1.0
            else:
                signals.iloc[i] = 0.0
                
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals