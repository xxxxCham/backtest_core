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
            "aroon_period": ParameterSpec(param_name="aroon_period", param_type="int", min_value=5, max_value=30, step=1),
            "sma_period": ParameterSpec(param_name="sma_period", param_type="int", min_value=10, max_value=50, step=1),
            "stop_atr_mult": ParameterSpec(param_name="stop_atr_mult", param_type="float", min_value=1.0, max_value=5.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_name="tp_atr_mult", param_type="float", min_value=2.0, max_value=10.0, step=0.5),
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
        
        sma = np.nan_to_num(indicators["sma"])
        aroon_up = np.nan_to_num(indicators["aroon"]["aroon_up"])
        aroon_down = np.nan_to_num(indicators["aroon"]["aroon_down"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Entry long conditions
        entry_long = (df["close"] > sma) & (aroon_up > aroon_down)
        
        # Exit conditions
        exit_long = (df["close"] < sma) | (aroon_up < aroon_down)
        
        # Initialize position
        position = 0
        
        # Generate signals
        for i in range(warmup, len(df)):
            if position == 0 and entry_long.iloc[i]:
                position = 1
                signals.iloc[i] = 1.0
            elif position == 1 and exit_long.iloc[i]:
                position = 0
                signals.iloc[i] = 0.0
        
        # Set warmup period
        signals.iloc[:warmup] = 0.0
        
        return signals