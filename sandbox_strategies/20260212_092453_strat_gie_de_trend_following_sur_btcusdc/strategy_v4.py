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
        return {"adx_threshold": 25, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_threshold": ParameterSpec(10, 50, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 6.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1),
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
        atr_values = np.nan_to_num(indicators["atr"])
        
        # Extract parameters
        adx_threshold = params.get("adx_threshold", 25)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))
        
        # Initialize entry and exit conditions
        entry_long = (supertrend_direction > 0) & (adx_value > adx_threshold)
        exit_signal = (supertrend_direction < 0) | (adx_value < adx_threshold)
        
        # Set warmup period
        signals.iloc[:warmup] = 0.0
        
        # Generate signals
        in_position = False
        position_entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        
        for i in range(warmup, len(signals)):
            if not in_position and entry_long[i]:
                signals[i] = 1.0
                in_position = True
                position_entry_price = df["close"].iloc[i]
                stop_loss = position_entry_price - (atr_values[i] * stop_atr_mult)
                take_profit = position_entry_price + (atr_values[i] * tp_atr_mult)
            elif in_position:
                current_price = df["close"].iloc[i]
                if current_price <= stop_loss or current_price >= take_profit or exit_signal[i]:
                    signals[i] = 0.0
                    in_position = False
                else:
                    signals[i] = 1.0
            else:
                signals[i] = 0.0
        
        return signals