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
            "tp_atr_mult": ParameterSpec(2.0, 4.0, float, "Take Profit ATR Multiplier"),
            "warmup": ParameterSpec(20, 100, int, "Warmup Period"),
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

        rsi = np.nan_to_num(indicators["rsi"])
        bollinger = indicators["bollinger"]
        atr = np.nan_to_num(indicators["atr"])

        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)

        long_condition = (
            bollinger["lower"] >= df["close"]
            & (rsi <= rsi_oversold)
            & (rsi > np.roll(rsi, 1))  # RSI increasing
        )

        short_condition = (
            bollinger["upper"] <= df["close"]
            & (rsi >= rsi_overbought)
            & (rsi < np.roll(rsi, 1))  # RSI decreasing
        )

        signals[long_condition] = 1.0
        signals[short_condition] = -1.0

        return signals