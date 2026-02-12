from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="ema_rsi_bollinger_scalp")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(70, 60, 80, "Integer", "RSI Overbought Level"),
            "rsi_oversold": ParameterSpec(30, 20, 40, "Integer", "RSI Oversold Level"),
            "rsi_period": ParameterSpec(14, 5, 21, "Integer", "RSI Period"),
            "stop_atr_mult": ParameterSpec(1.5, 0.5, 3.0, "Float", "Stop Loss ATR Multiplier"),
            "tp_atr_mult": ParameterSpec(3.0, 1.5, 5.0, "Float", "Take Profit ATR Multiplier"),
            "warmup": ParameterSpec(50, 20, 100, "Integer", "Warmup Period")
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

        rsi_overbought = int(params.get("rsi_overbought", 70))
        rsi_oversold = int(params.get("rsi_oversold", 30))

        ema_val = np.nan_to_num(indicators["ema"])
        rsi_val = np.nan_to_num(indicators["rsi"])
        bollinger = indicators["bollinger"]
        upper_band = np.nan_to_num(bollinger["upper"])
        lower_band = np.nan_to_num(bollinger["lower"])
        atr_val = np.nan_to_num(indicators["atr"])

        long_condition = (df["close"] < ema_val) & (rsi_val < 50) & (rsi_val > rsi_oversold) & (df["close"] <= lower_band)
        short_condition = (df["close"] > ema_val) & (rsi_val > 50) & (rsi_val < rsi_overbought) & (df["close"] >= upper_band)

        signals[long_condition] = 1.0
        signals[short_condition] = -1.0

        return signals