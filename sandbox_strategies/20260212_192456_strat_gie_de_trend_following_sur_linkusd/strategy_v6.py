from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="supertrend_adx_atr_trend_following")

    @property
    def required_indicators(self) -> List[str]:
        return ["supertrend", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"adx_period": 14, "adx_threshold": 35, "stop_atr_mult": 1.0, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(10, 30, 1),
            "adx_threshold": ParameterSpec(20, 50, 1),
            "stop_atr_mult": ParameterSpec(0.5, 2.0, 0.1),
            "supertrend_multiplier": ParameterSpec(1.0, 5.0, 0.1),
            "supertrend_period": ParameterSpec(5, 20, 1),
            "tp_atr_mult": ParameterSpec(1.0, 5.0, 0.1),
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
        adx_value = np.nan_to_num(indicators["adx"]["adx"])
        adx_threshold = params.get("adx_threshold", 35)
        supertrend_line = np.nan_to_num(indicators["supertrend"]["supertrend"])
        supertrend_direction = np.nan_to_num(indicators["supertrend"]["direction"])
        atr_value = np.nan_to_num(indicators["atr"])
        stop_atr_mult = params.get("stop_atr_mult", 1.0)
        tp_atr_mult = params.get("tp_atr_mult", 2.0)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        long_condition = (supertrend_line < close) & (supertrend_direction > 0) & (adx_value > adx_threshold)
        signals.loc[long_condition] = 1.0
        return signals