from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mean_reversion_btcusdc_short")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "stoch_rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(10, 30, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        bb = indicators["bollinger"]
        stoch_rsi = indicators["stoch_rsi"]
        atr = np.nan_to_num(indicators["atr"])
        
        lower = np.nan_to_num(bb["lower"])
        middle = np.nan_to_num(bb["middle"])
        k = np.nan_to_num(stoch_rsi["k"])
        d = np.nan_to_num(stoch_rsi["d"])
        
        # Short entry conditions
        entry_condition = (lower < middle) & (k > d) & (k > 80)
        
        # Exit condition
        exit_condition = lower > middle
        
        # Generate signals
        short_entries = entry_condition
        short_exits = exit_condition
        
        # Initialize signals to 0.0
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Set short signals
        signals[short_entries] = -1.0
        signals[short_exits] = 0.0
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals