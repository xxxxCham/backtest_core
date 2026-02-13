from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_trend_following_30m")

    @property
    def required_indicators(self) -> List[str]:
        return ["sma", "aroon", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"aroon_period": 14, "sma_fast": 50, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "aroon_period": ParameterSpec("aroon_period", 5, 30, 1),
            "sma_fast": ParameterSpec("sma_fast", 20, 100, 5),
            "stop_atr_mult": ParameterSpec("stop_atr_mult", 1.0, 5.0, 0.5),
            "tp_atr_mult": ParameterSpec("tp_atr_mult", 3.0, 10.0, 0.5),
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
        sma_fast = int(params.get("sma_fast", 50))
        stop_atr_mult = float(params.get("stop_atr_mult", 2.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 5.0))
        warmup = int(params.get("warmup", 50))
        
        sma = np.nan_to_num(indicators["sma"])
        aroon = indicators["aroon"]
        aroon_up = np.nan_to_num(aroon["aroon_up"])
        aroon_down = np.nan_to_num(aroon["aroon_down"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Calculate SMA slope
        sma_slope = np.diff(sma, prepend=sma[0])
        
        # Entry conditions
        entry_long = (sma_slope > 0) & (aroon_up > aroon_down)
        
        # Exit conditions
        exit_long = (sma_slope <= 0) | (aroon_up < aroon_down)
        
        # Generate signals
        positions = pd.Series(0.0, index=df.index)
        in_position = False
        
        for i in range(len(df)):
            if not in_position and entry_long[i]:
                positions[i] = 1.0
                in_position = True
            elif in_position and exit_long[i]:
                positions[i] = 0.0
                in_position = False
            elif in_position:
                positions[i] = 1.0
        
        signals = positions
        
        # Warmup protection
        signals.iloc[:warmup] = 0.0
        
        return signals