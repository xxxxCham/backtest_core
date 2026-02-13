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
        return {"aroon_period": 14, "aroon_threshold": 70, "ema_fast": 20, "ema_slow": 50, "stop_atr_mult": 1.0, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "aroon_period": ParameterSpec("aroon_period", 5, 30, 1),
            "aroon_threshold": ParameterSpec("aroon_threshold", 50, 90, 5),
            "ema_fast": ParameterSpec("ema_fast", 10, 50, 5),
            "ema_slow": ParameterSpec("ema_slow", 30, 100, 5),
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
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        aroon_period = int(params.get("aroon_period", 14))
        aroon_threshold = float(params.get("aroon_threshold", 70))
        ema_fast = int(params.get("ema_fast", 20))
        ema_slow = int(params.get("ema_slow", 50))
        stop_atr_mult = float(params.get("stop_atr_mult", 1.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.0))
        
        ema_fast_vals = np.nan_to_num(indicators["ema"])
        ema_slow_vals = np.nan_to_num(indicators["ema"])
        aroon_up = np.nan_to_num(indicators["aroon"]["aroon_up"])
        aroon_down = np.nan_to_num(indicators["aroon"]["aroon_down"])
        atr_vals = np.nan_to_num(indicators["atr"])
        
        # EMA trend confirmation
        ema_fast_line = ema_fast_vals[ema_fast - 1]
        ema_slow_line = ema_slow_vals[ema_slow - 1]
        trend_up = ema_fast_line > ema_slow_line
        
        # AROON momentum validation
        aroon_momentum = aroon_up > aroon_down
        aroon_strength = aroon_up > aroon_threshold
        
        # Entry condition
        entry_condition = trend_up & aroon_momentum & aroon_strength
        
        # Exit condition
        exit_condition = (ema_fast_line < ema_slow_line) | (aroon_up < aroon_down) | (aroon_up < 30)
        
        # Generate signals
        long_entries = np.where(entry_condition, 1.0, 0.0)
        long_exits = np.where(exit_condition, 0.0, 1.0)
        
        signals = pd.Series(long_entries * long_exits, index=df.index, dtype=np.float64)
        
        return signals