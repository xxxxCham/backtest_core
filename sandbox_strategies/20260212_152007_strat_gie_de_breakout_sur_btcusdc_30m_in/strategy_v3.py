from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="BTCUSDC_30m_Breakout_Strategy")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "supertrend", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"atr_period": 14, "keltner_multiplier": 1.5, "keltner_period": 20, "stop_atr_mult": 1.5, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 3.5}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "atr_period": ParameterSpec(param_type="int", min_value=5, max_value=30, step=1),
            "keltner_multiplier": ParameterSpec(param_type="float", min_value=0.5, max_value=3.0, step=0.1),
            "keltner_period": ParameterSpec(param_type="int", min_value=10, max_value=50, step=1),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=3.0, step=0.1),
            "supertrend_multiplier": ParameterSpec(param_type="float", min_value=1.0, max_value=5.0, step=0.1),
            "supertrend_period": ParameterSpec(param_type="int", min_value=5, max_value=20, step=1),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=6.0, step=0.1),
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
        supertrend = indicators["supertrend"]
        atr = np.nan_to_num(indicators["atr"])
        
        # Keltner bands
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_lower = np.nan_to_num(keltner["lower"])
        
        # Supertrend
        supertrend_value = np.nan_to_num(supertrend["supertrend"])
        supertrend_direction = np.nan_to_num(supertrend["direction"])
        
        # Close price
        close = np.nan_to_num(df["close"].values)
        
        # Entry conditions for short
        # Close crosses below Keltner lower band
        cross_below = (close < keltner_lower) & (np.roll(close, 1) >= np.roll(keltner_lower, 1))
        # Supertrend is bearish (direction = -1)
        trend_bearish = (supertrend_direction == -1)
        
        # Combine conditions
        entry_condition = cross_below & trend_bearish
        
        # Generate signals
        short_entries = np.where(entry_condition, -1.0, 0.0)
        signals = pd.Series(short_entries, index=df.index, dtype=np.float64)
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals