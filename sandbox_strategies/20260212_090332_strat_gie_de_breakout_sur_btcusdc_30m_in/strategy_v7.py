from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="keltner_supertrend_atr_breakout_enhanced")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "supertrend", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"keltner_multiplier": 1.5, "keltner_period": 20, "stop_atr_mult": 2.0, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "keltner_multiplier": ParameterSpec(1.0, 3.0, 0.1),
            "keltner_period": ParameterSpec(10, 50, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.1),
            "supertrend_multiplier": ParameterSpec(1.0, 5.0, 0.1),
            "supertrend_period": ParameterSpec(5, 20, 1),
            "tp_atr_mult": ParameterSpec(3.0, 10.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        close = np.nan_to_num(df['close'].values)
        keltner_mult = params.get("keltner_multiplier", 1.5)
        keltner_period = params.get("keltner_period", 20)
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        supertrend_mult = params.get("supertrend_multiplier", 3.0)
        supertrend_period = params.get("supertrend_period", 10)
        tp_atr_mult = params.get("tp_atr_mult", 5.0)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        keltner = indicators["keltner"]
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_lower = np.nan_to_num(keltner["lower"])
        atr = np.nan_to_num(indicators["atr"])
        supertrend = indicators["supertrend"]
        supertrend_direction = np.nan_to_num(supertrend["direction"])
        
        # Entry conditions
        long_entry = (close > keltner_upper) & (supertrend_direction > 0)
        short_entry = (close < keltner_lower) & (supertrend_direction < 0)
        
        # Initialize position tracking
        position = 0
        entry_price = 0.0
        entry_time = 0
        
        # Generate signals
        for i in range(warmup, len(signals)):
            if position == 0:
                if long_entry[i]:
                    signals[i] = 1.0
                    position = 1
                    entry_price = close[i]
                    entry_time = i
                elif short_entry[i]:
                    signals[i] = -1.0
                    position = -1
                    entry_price = close[i]
                    entry_time = i
            else:
                # Exit conditions
                take_profit = entry_price + tp_atr_mult * atr[i]
                stop_loss = entry_price - stop_atr_mult * atr[i]
                
                if position == 1:
                    # Long exit conditions
                    if close[i] <= keltner_lower[i] or close[i] <= stop_loss or close[i] >= take_profit:
                        signals[i] = 0.0
                        position = 0
                elif position == -1:
                    # Short exit conditions
                    if close[i] >= keltner_upper[i] or close[i] >= stop_loss or close[i] <= take_profit:
                        signals[i] = 0.0
                        position = 0
                        
        return signals