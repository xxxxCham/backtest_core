from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="trend_rsi_bollinger")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            # fill each tunable parameter
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        warmup = int(params.get("warmup", 50))
        
        # Extract necessary data
        close_price = df['close'].values
        rsi_values = np.nan_to_num(indicators["rsi"])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        atr_values = np.nan_to_num(indicators["atr"])
        
        # Calculate entry conditions
        long_entry = (close_price > bb_upper) & (rsi_values < params['rsi_oversold'])
        short_entry = (close_price < bb_lower) & (rsi_values > params['rsi_overbought'])
        
        # Apply warmup period protection
        signals.iloc[:warmup] = 0.0
        
        # Generate signals
        signals[long_entry] = 1.0  # LONG signal
        signals[short_entry] = -1.0  # SHORT signal
        
        return signals