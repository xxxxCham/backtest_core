from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_momentum_three_indicator")

    @property
    def required_indicators(self) -> List[str]:
        return ["roc", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"roc_period": 10, "rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 2.0, "tp_atr_mult": 4.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "roc_period": ParameterSpec(param_type="int", min_value=5, max_value=20, step=1),
            "rsi_overbought": ParameterSpec(param_type="int", min_value=60, max_value=80, step=5),
            "rsi_oversold": ParameterSpec(param_type="int", min_value=20, max_value=40, step=5),
            "rsi_period": ParameterSpec(param_type="int", min_value=10, max_value=20, step=1),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=3.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=3.0, max_value=6.0, step=0.5),
            "warmup": ParameterSpec(param_type="int", min_value=30, max_value=100, step=10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        roc_period = int(params.get("roc_period", 10))
        rsi_overbought = float(params.get("rsi_overbought", 70))
        rsi_oversold = float(params.get("rsi_oversold", 30))
        rsi_period = int(params.get("rsi_period", 14))
        stop_atr_mult = float(params.get("stop_atr_mult", 2.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 4.5))
        warmup = int(params.get("warmup", 50))
        
        roc = np.nan_to_num(indicators["roc"])
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Entry conditions
        long_condition = (roc > 0) & (rsi > rsi_oversold) & (roc > np.roll(roc, 1))
        short_condition = (roc < 0) & (rsi < rsi_overbought) & (roc < np.roll(roc, 1))
        
        # Exit conditions
        exit_long_condition = (roc < 0) & (rsi < rsi_oversold)
        exit_short_condition = (roc > 0) & (rsi > rsi_overbought)
        
        # Generate signals
        long_signals = long_condition & ~exit_long_condition
        short_signals = short_condition & ~exit_short_condition
        
        # Convert to signals
        signals[long_signals] = 1.0
        signals[short_signals] = -1.0
        
        # Warmup protection
        signals.iloc[:warmup] = 0.0
        
        return signals