from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_scalp")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(70, 90, float, "RSI Overbought Level"),
            "rsi_oversold": ParameterSpec(20, 30, float, "RSI Oversold Level"),
            "rsi_period": ParameterSpec(10, 20, int, "RSI Period"),
            "stop_atr_mult": ParameterSpec(1.0, 2.0, float, "Stop Loss ATR Multiplier"),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, float, "Take Profit ATR Multiplier"),
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

        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_period = params.get("rsi_period", 14)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)

        rsi_values = np.nan_to_num(indicators["rsi"])
        bollinger = indicators["bollinger"]
        atr_values = np.nan_to_num(indicators["atr"])

        long_condition = (
            (bollinger["lower"].astype(bool)) &
            (rsi_values > rsi_oversold) &
            (rsi_values < 50)
        )
        short_condition = (
            (bollinger["upper"].astype(bool)) &
            (rsi_values < rsi_overbought) &
            (rsi_values > 50)
        )

        signals[long_condition] = 1.0
        signals[short_condition] = -1.0

        return signals