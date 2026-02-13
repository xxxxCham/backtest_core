from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="supertrend_adx_trend_following")

    @property
    def required_indicators(self) -> List[str]:
        return ["supertrend", "adx", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"adx_period": 14, "adx_threshold": 25.0, "rsi_period": 14, "stop_atr_mult": 1.5, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(10, 30, 1),
            "adx_threshold": ParameterSpec(15.0, 30.0, 1.0),
            "rsi_period": ParameterSpec(10, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "supertrend_multiplier": ParameterSpec(1.0, 5.0, 0.1),
            "supertrend_period": ParameterSpec(5, 20, 1),
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
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Extract indicators
        supertrend_direction = np.nan_to_num(indicators["supertrend"]["direction"])
        adx_value = np.nan_to_num(indicators["adx"]["adx"])
        rsi_value = np.nan_to_num(indicators["rsi"])
        atr_value = np.nan_to_num(indicators["atr"])
        
        # Entry condition: supertrend up, adx strong, rsi not overbought
        entry_condition = (supertrend_direction > 0) & (adx_value >= params["adx_threshold"]) & (rsi_value < 50.0)
        
        # Exit condition: supertrend down, adx weak, rsi overbought
        exit_condition = (supertrend_direction < 0) | (adx_value < 20.0) | (rsi_value > 50.0)
        
        # Generate signals
        long_entries = entry_condition & (np.roll(signals, 1) != 1.0)
        long_exits = exit_condition & (np.roll(signals, 1) == 1.0)
        
        signals.loc[long_entries] = 1.0
        signals.loc[long_exits] = 0.0
        
        # Ensure no conflicting signals in same bar
        for i in range(1, len(signals)):
            if signals.iloc[i] == 1.0 and signals.iloc[i-1] == 1.0:
                signals.iloc[i] = 0.0
            elif signals.iloc[i] == 0.0 and signals.iloc[i-1] == 0.0:
                signals.iloc[i] = 0.0
                
        return signals