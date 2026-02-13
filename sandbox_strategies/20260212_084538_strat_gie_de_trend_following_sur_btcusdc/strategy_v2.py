from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_trend_following_sma_adx_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["sma", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"adx_threshold": 20, "sma_fast": 5, "sma_slow": 20, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_threshold": ParameterSpec(10, 30, 1),
            "sma_fast": ParameterSpec(2, 10, 1),
            "sma_slow": ParameterSpec(15, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.5),
            "tp_atr_mult": ParameterSpec(3.0, 8.0, 0.5),
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
        
        adx_threshold = params.get("adx_threshold", 20)
        sma_fast = params.get("sma_fast", 5)
        sma_slow = params.get("sma_slow", 20)
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 5.0)
        
        # Get indicator values
        sma_fast_vals = np.nan_to_num(indicators["sma"][sma_fast])
        sma_slow_vals = np.nan_to_num(indicators["sma"][sma_slow])
        adx_vals = np.nan_to_num(indicators["adx"]["adx"])
        atr_vals = np.nan_to_num(indicators["atr"])
        
        # Entry condition: SMA crossover with ADX filter
        entry_condition = (sma_fast_vals > sma_slow_vals) & (adx_vals > adx_threshold)
        
        # Exit condition: SMA crossunder with ADX filter
        exit_condition = (sma_fast_vals < sma_slow_vals) & (adx_vals < adx_threshold)
        
        # Generate signals
        positions = np.zeros_like(entry_condition, dtype=np.float64)
        in_position = False
        current_position = 0
        
        for i in range(len(entry_condition)):
            if not in_position and entry_condition[i]:
                positions[i] = 1.0
                in_position = True
                current_position = 1.0
            elif in_position and exit_condition[i]:
                positions[i] = 0.0
                in_position = False
                current_position = 0.0
            elif in_position:
                positions[i] = 1.0
            else:
                positions[i] = 0.0
                
        signals = pd.Series(positions, index=df.index, dtype=np.float64)
        return signals