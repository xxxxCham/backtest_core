from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="keltner_supertrend_atr_breakout")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "supertrend", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"atr_period": 14, "keltner_multiplier": 1.5, "keltner_period": 20, "stop_atr_mult": 2.0, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "atr_period": ParameterSpec(10, 30, 1),
            "keltner_multiplier": ParameterSpec(1.0, 3.0, 0.1),
            "keltner_period": ParameterSpec(10, 50, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.1),
            "supertrend_multiplier": ParameterSpec(1.0, 5.0, 0.1),
            "supertrend_period": ParameterSpec(5, 20, 1),
            "tp_atr_mult": ParameterSpec(2.0, 10.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        keltner = indicators["keltner"]
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_lower = np.nan_to_num(keltner["lower"])
        keltner_middle = np.nan_to_num(keltner["middle"])
        
        supertrend = indicators["supertrend"]
        supertrend_value = np.nan_to_num(supertrend["supertrend"])
        supertrend_direction = np.nan_to_num(supertrend["direction"])
        
        atr = np.nan_to_num(indicators["atr"])
        
        # Extract close prices
        close = np.nan_to_num(df["close"].values)
        
        # Warmup period
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Entry conditions
        # Long entry: close breaks above Keltner upper, supertrend direction is positive, and close is increasing
        long_condition = (close > keltner_upper) & (supertrend_direction > 0) & (close > np.roll(close, 1))
        
        # Short entry: close breaks below Keltner lower, supertrend direction is negative, and close is decreasing
        short_condition = (close < keltner_lower) & (supertrend_direction < 0) & (close < np.roll(close, 1))
        
        # Exit conditions
        # Exit on supertrend crossover
        supertrend_cross = (np.roll(supertrend_value, 1) <= np.roll(close, 1)) & (supertrend_value > close)
        supertrend_cross_short = (np.roll(supertrend_value, 1) >= np.roll(close, 1)) & (supertrend_value < close)
        
        # Exit on Keltner middle crossover
        keltner_middle_cross = (np.roll(keltner_middle, 1) <= np.roll(close, 1)) & (keltner_middle > close)
        keltner_middle_cross_short = (np.roll(keltner_middle, 1) >= np.roll(close, 1)) & (keltner_middle < close)
        
        # Combine exit conditions
        exit_long = supertrend_cross | keltner_middle_cross
        exit_short = supertrend_cross_short | keltner_middle_cross_short
        
        # Generate signals
        long_signals = np.where(long_condition, 1.0, 0.0)
        short_signals = np.where(short_condition, -1.0, 0.0)
        
        # Apply exit conditions
        for i in range(1, len(signals)):
            if signals.iloc[i-1] == 1.0 and exit_long[i]:
                signals.iloc[i] = 0.0
            elif signals.iloc[i-1] == -1.0 and exit_short[i]:
                signals.iloc[i] = 0.0
            elif long_signals[i] == 1.0:
                signals.iloc[i] = 1.0
            elif short_signals[i] == -1.0:
                signals.iloc[i] = -1.0
                
        return signals