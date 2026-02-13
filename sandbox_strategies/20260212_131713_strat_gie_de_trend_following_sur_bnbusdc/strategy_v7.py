from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="aroon_ema_trend_following")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "aroon", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"aroon_period": 14, "aroon_threshold": 70, "ema_fast": 10, "ema_slow": 20, "stop_atr_mult": 1.0, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "aroon_period": ParameterSpec("aroon_period", 5, 30, 1),
            "aroon_threshold": ParameterSpec("aroon_threshold", 50, 90, 5),
            "ema_fast": ParameterSpec("ema_fast", 5, 20, 1),
            "ema_slow": ParameterSpec("ema_slow", 15, 50, 1),
            "stop_atr_mult": ParameterSpec("stop_atr_mult", 0.5, 2.0, 0.5),
            "tp_atr_mult": ParameterSpec("tp_atr_mult", 1.0, 4.0, 0.5),
            "warmup": ParameterSpec("warmup", 20, 100, 10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        aroon_period = int(params.get("aroon_period", 14))
        aroon_threshold = float(params.get("aroon_threshold", 70))
        ema_fast = int(params.get("ema_fast", 10))
        ema_slow = int(params.get("ema_slow", 20))
        stop_atr_mult = float(params.get("stop_atr_mult", 1.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.0))
        warmup = int(params.get("warmup", 50))
        
        ema_fast_vals = np.nan_to_num(indicators["ema"][ema_fast])
        ema_slow_vals = np.nan_to_num(indicators["ema"][ema_slow])
        aroon_up = np.nan_to_num(indicators["aroon"]["aroon_up"])
        aroon_down = np.nan_to_num(indicators["aroon"]["aroon_down"])
        atr_vals = np.nan_to_num(indicators["atr"])
        
        # Entry conditions
        entry_long = (ema_fast_vals > ema_slow_vals) & (aroon_up > aroon_down) & (aroon_up > aroon_threshold)
        
        # Exit conditions
        exit_long = (ema_fast_vals < ema_slow_vals) | (aroon_up < aroon_down) | (aroon_up < 30)
        
        # Generate signals
        entry_mask = entry_long
        exit_mask = exit_long
        
        # Initialize signal array
        signal_values = np.zeros(len(df))
        
        # Set signals
        for i in range(len(df)):
            if entry_mask[i] and i > warmup:
                signal_values[i] = 1.0  # LONG
            elif exit_mask[i] and i > warmup:
                signal_values[i] = 0.0  # FLAT
        
        signals.iloc[warmup:] = signal_values[warmup:]
        
        return signals