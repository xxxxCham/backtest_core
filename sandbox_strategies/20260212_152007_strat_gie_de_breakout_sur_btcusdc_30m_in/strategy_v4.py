from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_30m_breakout_short")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "supertrend", "momentum", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"keltner_multiplier": 1.5, "keltner_period": 20, "momentum_period": 10, "stop_atr_mult": 1.5, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 3.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "keltner_multiplier": ParameterSpec(1.0, 3.0, 0.1),
            "keltner_period": ParameterSpec(10, 50, 1),
            "momentum_period": ParameterSpec(5, 20, 1),
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
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Extract indicators
        keltner = indicators["keltner"]
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_lower = np.nan_to_num(keltner["lower"])
        supertrend = indicators["supertrend"]
        supertrend_direction = np.nan_to_num(supertrend["direction"])
        momentum = np.nan_to_num(indicators["momentum"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Entry condition: price breaks below Keltner lower band, Supertrend is bearish, momentum is negative
        entry_condition = (df["close"] < keltner_lower) & (supertrend_direction == -1) & (momentum < 0)
        
        # Exit condition: price re-enters Keltner upper band
        exit_condition = df["close"] > keltner_upper
        
        # Generate signals
        signals[entry_condition] = -1.0
        signals[exit_condition] = 0.0
        
        return signals