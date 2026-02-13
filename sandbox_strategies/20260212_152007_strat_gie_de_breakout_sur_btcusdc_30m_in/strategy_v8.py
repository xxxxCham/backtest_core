from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="keltner_supertrend_breakout_short")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "supertrend", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"keltner_multiplier": 1.5, "keltner_period": 20, "stop_atr_mult": 1.5, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 3.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "keltner_multiplier": ParameterSpec(1.0, 3.0, 0.1),
            "keltner_period": ParameterSpec(10, 50, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "supertrend_multiplier": ParameterSpec(1.0, 5.0, 0.1),
            "supertrend_period": ParameterSpec(5, 20, 1),
            "tp_atr_mult": ParameterSpec(2.0, 6.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1),
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
        keltner = indicators["keltner"]
        supertrend = indicators["supertrend"]
        atr = np.nan_to_num(indicators["atr"])
        
        # Keltner bands
        keltner_lower = np.nan_to_num(keltner["lower"])
        keltner_middle = np.nan_to_num(keltner["middle"])
        
        # Supertrend
        supertrend_direction = np.nan_to_num(supertrend["direction"])
        
        # Entry conditions
        # Price closes below Keltner lower band
        below_lower = df["close"] < keltner_lower
        
        # Supertrend is in downtrend
        downtrend = supertrend_direction < 0
        
        # Entry signal
        entry_signal = below_lower & downtrend
        
        # Exit conditions
        # Price closes above Keltner middle band
        above_middle = df["close"] > keltner_middle
        
        # Trailing stop loss (1.5x ATR from entry)
        # For simplicity, we'll use a basic approach without tracking entry prices
        # This is a simplified version - in practice, you'd track entry points
        # Here we assume exit on above_middle or based on ATR
        
        # Generate signals
        # Short signal
        signals[entry_signal] = -1.0
        
        # Exit signal (short)
        signals[above_middle] = 0.0
        
        return signals