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
        return {"adx_period": 14, "adx_threshold": 25.0, "stop_atr_mult": 1.0, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(param_type="int", min_value=5, max_value=30, step=1),
            "adx_threshold": ParameterSpec(param_type="float", min_value=10.0, max_value=50.0, step=1.0),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=0.5, max_value=3.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=5.0, step=0.5),
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
        
        # Entry condition: supertrend direction > 0 AND adx >= threshold
        entry_condition = (supertrend_direction > 0) & (adx_value >= params["adx_threshold"])
        
        # Exit condition: supertrend direction < 0 OR adx < threshold
        exit_condition = (supertrend_direction < 0) | (adx_value < params["adx_threshold"])
        
        # Generate signals
        positions = np.zeros_like(supertrend_direction)
        in_position = False
        entry_price = 0.0
        
        for i in range(len(supertrend_direction)):
            if entry_condition[i] and not in_position:
                positions[i] = 1.0  # LONG
                in_position = True
                entry_price = df["close"].iloc[i]
            elif exit_condition[i] and in_position:
                positions[i] = 0.0  # FLAT
                in_position = False
                entry_price = 0.0
            elif in_position:
                positions[i] = 1.0  # Continue holding LONG
        
        signals = pd.Series(positions, index=df.index, dtype=np.float64)
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals