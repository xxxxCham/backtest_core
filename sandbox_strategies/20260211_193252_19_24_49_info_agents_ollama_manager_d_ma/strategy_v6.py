from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="ema_bollinger_rsi_scalper_v4")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "bollinger", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 90, 1),
            "rsi_oversold": ParameterSpec(10, 40, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        ema_21 = np.nan_to_num(indicators["ema"])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Warmup period
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Entry conditions
        long_condition = (df["close"] < ema_21) & (df["close"] > bb_lower) & (rsi > params["rsi_oversold"]) & (df["close"] > df["open"])
        short_condition = (df["close"] > ema_21) & (df["close"] < bb_upper) & (rsi < params["rsi_overbought"]) & (df["close"] < df["open"])
        
        # Exit conditions
        long_exit = (df["close"] > bb_upper) | (rsi > params["rsi_overbought"])
        short_exit = (df["close"] < bb_lower) | (rsi < params["rsi_oversold"])
        
        # Set signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0
        
        return signals