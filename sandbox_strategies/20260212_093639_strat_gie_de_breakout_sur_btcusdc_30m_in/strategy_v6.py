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
        return {"keltner_multiplier": 1.5, "keltner_period": 20, "stop_atr_mult": 2.0, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Extract indicators
        keltner = indicators["keltner"]
        lower_keltner = np.nan_to_num(keltner["lower"])
        upper_keltner = np.nan_to_num(keltner["upper"])
        supertrend = indicators["supertrend"]
        st_direction = np.nan_to_num(supertrend["direction"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Entry conditions for short
        price = np.nan_to_num(df["close"].values)
        price_breaks_below_lower = price < lower_keltner
        downtrend = st_direction < 0
        
        # Entry signal
        entry_condition = price_breaks_below_lower & downtrend
        
        # Exit conditions
        price_closes_above_upper = price > upper_keltner
        
        # Initialize exit mask
        exit_mask = np.zeros_like(entry_condition, dtype=bool)
        
        # Simple trailing stop logic (simplified for this implementation)
        # In a real system, you'd track the entry price and calculate trailing stop
        # Here, we just use a simple logic based on price crossing upper channel
        exit_mask = price_closes_above_upper
        
        # Generate signals
        # Short entry
        short_entry = pd.Series(0.0, index=df.index)
        short_entry[entry_condition] = -1.0
        
        # Exit signal
        short_exit = pd.Series(0.0, index=df.index)
        short_exit[exit_mask] = 0.0
        
        # Combine signals
        signals = short_entry + short_exit
        signals = signals.clip(-1.0, 1.0)
        
        # Ensure no early signals
        signals.iloc[:warmup] = 0.0
        
        return signals