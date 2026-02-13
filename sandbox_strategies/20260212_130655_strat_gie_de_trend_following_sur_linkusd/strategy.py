from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="aroon_sma_trend_following")

    @property
    def required_indicators(self) -> List[str]:
        return ["sma", "aroon", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"aroon_period": 14, "sma_period": 20, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "aroon_period": ParameterSpec(param_name="aroon_period", param_type="int", min_value=5, max_value=30, step=1),
            "sma_period": ParameterSpec(param_name="sma_period", param_type="int", min_value=10, max_value=50, step=5),
            "stop_atr_mult": ParameterSpec(param_name="stop_atr_mult", param_type="float", min_value=1.0, max_value=3.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_name="tp_atr_mult", param_type="float", min_value=2.0, max_value=6.0, step=0.5),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        aroon_period = int(params.get("aroon_period", 14))
        sma_period = int(params.get("sma_period", 20))
        stop_atr_mult = float(params.get("stop_atr_mult", 1.5))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.0))
        warmup = int(params.get("warmup", 50))
        
        sma = np.nan_to_num(indicators["sma"])
        aroon = indicators["aroon"]
        aroon_up = np.nan_to_num(aroon["aroon_up"])
        aroon_down = np.nan_to_num(aroon["aroon_down"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Calculate SMA slope
        sma_shifted = np.roll(sma, 1)
        sma_slope = sma - sma_shifted
        
        # Entry conditions
        entry_long = (sma_slope > 0) & (aroon_up > aroon_down) & (aroon_up > 70)
        
        # Exit condition
        exit_signal = (sma_slope < 0) | (aroon_down > aroon_up)
        
        # Generate signals
        position = 0
        for i in range(len(df)):
            if i < warmup:
                signals.iloc[i] = 0.0
                continue
                
            if entry_long[i] and position == 0:
                position = 1
                signals.iloc[i] = 1.0
            elif exit_signal[i] and position == 1:
                position = 0
                signals.iloc[i] = 0.0
            elif position == 1:
                signals.iloc[i] = 1.0
            else:
                signals.iloc[i] = 0.0
                
        return signals