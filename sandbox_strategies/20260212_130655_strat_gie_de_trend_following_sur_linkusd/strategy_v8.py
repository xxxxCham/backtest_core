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
        return {"aroon_period": 14, "sma_period": 20, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "aroon_period": ParameterSpec("aroon_period", 5, 30, 1),
            "sma_period": ParameterSpec("sma_period", 5, 50, 1),
            "stop_atr_mult": ParameterSpec("stop_atr_mult", 1.0, 5.0, 0.5),
            "tp_atr_mult": ParameterSpec("tp_atr_mult", 2.0, 10.0, 0.5),
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
        stop_atr_mult = float(params.get("stop_atr_mult", 2.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 5.0))
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        sma = indicators["sma"]
        aroon = indicators["aroon"]
        atr = indicators["atr"]
        
        sma_array = np.nan_to_num(sma)
        aroon_up = np.nan_to_num(aroon["aroon_up"])
        aroon_down = np.nan_to_num(aroon["aroon_down"])
        atr_array = np.nan_to_num(atr)
        
        # Entry long conditions
        sma_up = sma_array > np.roll(sma_array, 1)
        aroon_valid = aroon_up > aroon_down
        
        entry_long = sma_up & aroon_valid
        
        # Exit conditions
        sma_down = sma_array < np.roll(sma_array, 1)
        aroon_invalid = aroon_up < aroon_down
        
        exit_signal = sma_down | aroon_invalid
        
        # Generate signals
        positions = pd.Series(0.0, index=df.index)
        in_position = False
        for i in range(len(df)):
            if entry_long[i] and not in_position:
                positions.iloc[i] = 1.0
                in_position = True
            elif exit_signal[i] and in_position:
                positions.iloc[i] = 0.0
                in_position = False
            elif in_position:
                positions.iloc[i] = 1.0
            else:
                positions.iloc[i] = 0.0
                
        signals = positions
        return signals