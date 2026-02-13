from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="keltner_supertrend_breakout_v5")

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
            "warmup": ParameterSpec(30, 100, 1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        keltner_mult = params.get("keltner_multiplier", 1.5)
        keltner_period = params.get("keltner_period", 20)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        supertrend_mult = params.get("supertrend_multiplier", 3.0)
        supertrend_period = params.get("supertrend_period", 10)
        tp_atr_mult = params.get("tp_atr_mult", 3.5)
        warmup = int(params.get("warmup", 50))
        
        # Get indicators
        keltner = indicators["keltner"]
        upper_band = np.nan_to_num(keltner["upper"])
        middle_band = np.nan_to_num(keltner["middle"])
        lower_band = np.nan_to_num(keltner["lower"])
        
        supertrend = indicators["supertrend"]
        supertrend_value = np.nan_to_num(supertrend["supertrend"])
        direction = np.nan_to_num(supertrend["direction"])
        
        atr = np.nan_to_num(indicators["atr"])
        
        # Price
        close = np.nan_to_num(df["close"].values)
        
        # Entry conditions
        price_above_upper = close > upper_band
        price_below_lower = close < lower_band
        
        # Trend validation
        uptrend = direction > 0
        downtrend = direction < 0
        
        # Breakout confirmation
        long_condition = price_above_upper & uptrend
        short_condition = price_below_lower & downtrend
        
        # Generate signals
        signals.iloc[warmup:] = 0.0
        
        # Long entries
        long_entries = np.where(long_condition, 1.0, 0.0)
        signals.iloc[warmup:] = long_entries[warmup:]
        
        return signals