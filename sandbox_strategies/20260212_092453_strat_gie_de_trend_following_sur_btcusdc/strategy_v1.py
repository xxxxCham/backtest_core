from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="supertrend_adx_atr_trend_following")

    @property
    def required_indicators(self) -> List[str]:
        return ["supertrend", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"adx_period": 14, "adx_threshold": 20, "stop_atr_mult": 1.5, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(10, 30, 1),
            "adx_threshold": ParameterSpec(10, 40, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "supertrend_multiplier": ParameterSpec(1.0, 5.0, 0.1),
            "supertrend_period": ParameterSpec(5, 20, 1),
            "tp_atr_mult": ParameterSpec(2.0, 6.0, 0.1),
            "warmup": ParameterSpec(30, 100, 1)
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
        supertrend = indicators["supertrend"]
        adx = indicators["adx"]
        atr = indicators["atr"]
        
        # Process supertrend values
        supertrend_values = np.nan_to_num(supertrend["supertrend"])
        supertrend_direction = np.nan_to_num(supertrend["direction"])
        
        # Process ADX values
        adx_values = np.nan_to_num(adx["adx"])
        adx_threshold = params.get("adx_threshold", 20)
        
        # Process ATR values
        atr_values = np.nan_to_num(atr)
        
        # Entry conditions
        # Long entry: supertrend above price AND ADX > threshold
        price = np.nan_to_num(df["close"].values)
        entry_condition = (supertrend_direction > 0) & (adx_values > adx_threshold)
        
        # Exit conditions
        # Exit: supertrend below price OR ADX < threshold
        exit_condition = (supertrend_direction < 0) | (adx_values < adx_threshold)
        
        # Generate signals
        position = 0
        for i in range(len(signals)):
            if entry_condition[i] and position == 0:
                position = 1  # Long
                signals.iloc[i] = 1.0
            elif exit_condition[i] and position == 1:
                position = 0  # Flat
                signals.iloc[i] = 0.0
            elif position == 1:
                signals.iloc[i] = 1.0  # Hold long
            else:
                signals.iloc[i] = 0.0  # Flat
        
        return signals