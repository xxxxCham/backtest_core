from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_momentum_short_with_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "mfi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 2.0, "tp_atr_mult": 4.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec("rsi_overbought", 60, 80, 1),
            "rsi_oversold": ParameterSpec("rsi_oversold", 20, 40, 1),
            "rsi_period": ParameterSpec("rsi_period", 10, 20, 1),
            "stop_atr_mult": ParameterSpec("stop_atr_mult", 1.5, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec("tp_atr_mult", 3.0, 6.0, 0.1),
            "warmup": ParameterSpec("warmup", 30, 70, 1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        rsi = np.nan_to_num(indicators["rsi"])
        mfi = np.nan_to_num(indicators["mfi"])
        atr = np.nan_to_num(indicators["atr"])
        
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 4.5)
        warmup = int(params.get("warmup", 50))
        
        signals.iloc[:warmup] = 0.0
        
        short_condition = (rsi > rsi_overbought) & (mfi > 100 - rsi_overbought)
        signals[short_condition] = -1.0
        
        return signals