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
        return {"adx_period": 14, "adx_threshold": 30, "stop_atr_mult": 1.5, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 1.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(10, 30, 14),
            "adx_threshold": ParameterSpec(20, 40, 30),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 1.5),
            "supertrend_multiplier": ParameterSpec(1.0, 5.0, 3.0),
            "supertrend_period": ParameterSpec(5, 20, 10),
            "tp_atr_mult": ParameterSpec(1.0, 3.0, 1.5),
            "warmup": ParameterSpec(30, 100, 50),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        supertrend_line = np.nan_to_num(indicators["supertrend"]["supertrend"])
        supertrend_direction = np.nan_to_num(indicators["supertrend"]["direction"])
        adx_value = np.nan_to_num(indicators["adx"]["adx"])
        atr_value = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)
        
        # Entry conditions
        entry_condition = (
            (supertrend_line < close) &
            (supertrend_direction > 0) &
            (adx_value > params["adx_threshold"]) &
            (close > supertrend_line)
        )
        
        # Exit conditions
        exit_condition = (
            (supertrend_line > close) |
            (adx_value < 20)
        )
        
        # Generate signals
        long_positions = np.zeros_like(close, dtype=bool)
        in_position = False
        
        for i in range(len(close)):
            if entry_condition[i] and not in_position:
                long_positions[i] = True
                in_position = True
            elif exit_condition[i] and in_position:
                long_positions[i] = False
                in_position = False
        
        signals[long_positions] = 1.0
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals