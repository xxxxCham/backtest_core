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
        return {"adx_period": 14, "adx_threshold": 25.0, "stop_atr_mult": 1.5, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(param_type="int", min_value=5, max_value=30, step=1),
            "adx_threshold": ParameterSpec(param_type="float", min_value=10.0, max_value=40.0, step=1.0),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=3.0, step=0.1),
            "supertrend_multiplier": ParameterSpec(param_type="float", min_value=1.0, max_value=5.0, step=0.1),
            "supertrend_period": ParameterSpec(param_type="int", min_value=5, max_value=30, step=1),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=5.0, step=0.1),
            "warmup": ParameterSpec(param_type="int", min_value=20, max_value=100, step=10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Extract indicators
        supertrend = indicators["supertrend"]
        adx = indicators["adx"]
        atr = indicators["atr"]
        
        # Get supertrend direction and adx value
        supertrend_direction = np.nan_to_num(supertrend["direction"])
        adx_value = np.nan_to_num(adx["adx"])
        
        # Entry conditions
        entry_long = (supertrend_direction > 0) & (adx_value >= params["adx_threshold"])
        
        # Exit conditions
        exit_long = (supertrend_direction < 0) | (adx_value < 20.0)
        
        # Generate signals
        long_positions = np.zeros_like(supertrend_direction)
        in_position = False
        position_entry_index = -1
        
        for i in range(len(supertrend_direction)):
            if entry_long[i] and not in_position:
                long_positions[i] = 1.0
                in_position = True
                position_entry_index = i
            elif exit_long[i] and in_position:
                long_positions[i] = 0.0
                in_position = False
            elif in_position:
                long_positions[i] = 1.0
                
        signals = pd.Series(long_positions, index=df.index, dtype=np.float64)
        return signals