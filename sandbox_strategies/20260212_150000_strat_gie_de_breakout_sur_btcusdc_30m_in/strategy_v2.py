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
            "atr_period": ParameterSpec(5, 30, 1),
            "keltner_multiplier": ParameterSpec(0.5, 3.0, 0.1),
            "keltner_period": ParameterSpec(10, 50, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.1),
            "supertrend_multiplier": ParameterSpec(1.0, 5.0, 0.1),
            "supertrend_period": ParameterSpec(5, 30, 1),
            "tp_atr_mult": ParameterSpec(2.0, 10.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        close = np.nan_to_num(df["close"].values)
        atr_value = np.nan_to_num(indicators["atr"])
        keltner_upper = np.nan_to_num(indicators["keltner"]["upper"])
        keltner_lower = np.nan_to_num(indicators["keltner"]["lower"])
        supertrend_value = np.nan_to_num(indicators["supertrend"]["supertrend"])
        supertrend_direction = np.nan_to_num(indicators["supertrend"]["direction"])
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 5.0)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        long_condition = (close > keltner_upper) & (supertrend_direction > 0) & (close > np.roll(close, 1))
        short_condition = (close < keltner_lower) & (supertrend_direction < 0) & (close < np.roll(close, 1))
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0
        return signals