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
        return {"aroon_period": 14, "ema_fast": 20, "ema_slow": 50, "stop_atr_mult": 1.0, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "aroon_period": ParameterSpec(10, 30, 1),
            "ema_fast": ParameterSpec(5, 50, 1),
            "ema_slow": ParameterSpec(20, 100, 1),
            "stop_atr_mult": ParameterSpec(0.5, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec(1.0, 4.0, 0.1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        aroon_period = int(params.get("aroon_period", 14))
        ema_fast = int(params.get("ema_fast", 20))
        ema_slow = int(params.get("ema_slow", 50))
        stop_atr_mult = float(params.get("stop_atr_mult", 1.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.0))
        warmup = int(params.get("warmup", 50))
        
        ema_fast_line = np.nan_to_num(indicators["ema"][ema_fast])
        ema_slow_line = np.nan_to_num(indicators["ema"][ema_slow])
        aroon_up = np.nan_to_num(indicators["aroon"]["aroon_up"])
        aroon_down = np.nan_to_num(indicators["aroon"]["aroon_down"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Entry long: EMA fast > EMA slow AND AROON up > AROON down
        entry_long = (ema_fast_line > ema_slow_line) & (aroon_up > aroon_down)
        
        # Exit: EMA fast < EMA slow OR AROON up < AROON down
        exit_signal = (ema_fast_line < ema_slow_line) | (aroon_up < aroon_down)
        
        # Initialize position
        position = 0
        
        # Generate signals
        for i in range(len(df)):
            if position == 0 and entry_long[i]:
                position = 1
                signals.iloc[i] = 1.0
            elif position == 1 and exit_signal[i]:
                position = 0
                signals.iloc[i] = 0.0
        
        # Set warmup period
        signals.iloc[:warmup] = 0.0
        
        return signals