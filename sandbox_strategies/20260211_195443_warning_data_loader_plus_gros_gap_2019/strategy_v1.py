from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="rsi_bollinger_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(name="rsi_overbought", param_type="int", min_value=50, max_value=90, step=5),
            "rsi_oversold": ParameterSpec(name="rsi_oversold", param_type="int", min_value=10, max_value=50, step=5),
            "rsi_period": ParameterSpec(name="rsi_period", param_type="int", min_value=5, max_value=30, step=5),
            "stop_atr_mult": ParameterSpec(name="stop_atr_mult", param_type="float", min_value=1.0, max_value=3.0, step=0.5),
            "tp_atr_mult": ParameterSpec(name="tp_atr_mult", param_type="float", min_value=2.0, max_value=5.0, step=0.5),
            "warmup": ParameterSpec(name="warmup", param_type="int", min_value=20, max_value=100, step=10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        lower_band = np.nan_to_num(bb["lower"])
        upper_band = np.nan_to_num(bb["upper"])
        price = np.nan_to_num(df["close"].values)
        atr = np.nan_to_num(indicators["atr"])
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        long_conditions = (price < lower_band) & (rsi < rsi_oversold)
        short_conditions = (price > upper_band) & (rsi > rsi_overbought)
        long_signals = np.where(long_conditions, 1.0, 0.0)
        short_signals = np.where(short_conditions, -1.0, 0.0)
        signals = pd.Series(long_signals + short_signals, index=df.index, dtype=np.float64)
        return signals