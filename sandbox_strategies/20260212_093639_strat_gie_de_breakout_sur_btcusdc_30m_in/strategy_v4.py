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
        return {"atr_period": 14, "keltner_multiplier": 1.5, "keltner_period": 20, "stop_atr_mult": 2.0, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 5.0}

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
        
        # Price
        close = np.nan_to_num(df["close"].values)
        
        # Create conditions
        # Short entry: Price breaks above Keltner upper band AND Supertrend is trending upward
        entry_condition = (close > keltner_upper) & (supertrend_direction > 0)
        
        # Exit condition: Price closes below Keltner upper band OR trailing stop loss triggered
        exit_condition = close < keltner_upper
        
        # Generate signals
        entry_signals = pd.Series(0.0, index=df.index)
        entry_signals[entry_condition] = -1.0  # Short signal
        
        # Apply exit condition
        exit_signals = pd.Series(0.0, index=df.index)
        exit_signals[exit_condition] = 0.0  # Flat signal
        
        # Combine signals
        signals = entry_signals
        signals[exit_condition] = 0.0
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals