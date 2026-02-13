from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_sma_adx_atr_short")

    @property
    def required_indicators(self) -> List[str]:
        return ["sma", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"adx_period": 14, "adx_threshold": 25, "sma_fast": 5, "sma_slow": 20, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(param_name="adx_period", param_type="int", min_value=5, max_value=30, step=1),
            "adx_threshold": ParameterSpec(param_name="adx_threshold", param_type="int", min_value=10, max_value=50, step=5),
            "sma_fast": ParameterSpec(param_name="sma_fast", param_type="int", min_value=3, max_value=15, step=1),
            "sma_slow": ParameterSpec(param_name="sma_slow", param_type="int", min_value=10, max_value=50, step=5),
            "stop_atr_mult": ParameterSpec(param_name="stop_atr_mult", param_type="float", min_value=1.0, max_value=5.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_name="tp_atr_mult", param_type="float", min_value=3.0, max_value=10.0, step=0.5),
            "warmup": ParameterSpec(param_name="warmup", param_type="int", min_value=30, max_value=100, step=10),
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
        
        # Extract indicators
        sma_fast = np.nan_to_num(indicators["sma"])
        sma_slow = np.nan_to_num(indicators["sma"])
        adx_val = np.nan_to_num(indicators["adx"]["adx"])
        atr_val = np.nan_to_num(indicators["atr"])
        
        # Entry condition: short when fast SMA crosses below slow SMA and ADX is above threshold
        entry_condition = (sma_fast < sma_slow) & (adx_val > params["adx_threshold"])
        
        # Exit condition: when fast SMA crosses above slow SMA or ADX drops below threshold
        exit_condition = (sma_fast > sma_slow) | (adx_val < 20)
        
        # Generate signals
        entry_indices = np.where(entry_condition)[0]
        exit_indices = np.where(exit_condition)[0]
        
        # Set signals to -1.0 for short positions
        for idx in entry_indices:
            if idx >= warmup:
                signals.iloc[idx] = -1.0
                
        # Exit short positions
        for idx in exit_indices:
            if idx >= warmup and signals.iloc[idx] == -1.0:
                signals.iloc[idx] = 0.0
                
        return signals