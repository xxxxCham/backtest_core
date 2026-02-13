from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="momentum_avaxusdc_1d")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(50, 90, 1),
            "rsi_oversold": ParameterSpec(10, 50, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 6.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        rsi = np.nan_to_num(indicators["rsi"])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        atr = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)
        
        # Params
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        warmup = int(params.get("warmup", 50))
        
        # Entry conditions
        # Long entry
        rsi_long_condition = (rsi > rsi_oversold) & (np.roll(rsi, 1) <= rsi_oversold)
        price_long_condition = close > bb_upper
        long_condition = rsi_long_condition & price_long_condition
        
        # Short entry
        rsi_short_condition = (rsi < rsi_overbought) & (np.roll(rsi, 1) >= rsi_overbought)
        price_short_condition = close < bb_lower
        short_condition = rsi_short_condition & price_short_condition
        
        # Signal generation
        long_signals = np.zeros_like(rsi)
        short_signals = np.zeros_like(rsi)
        
        # Set long signals
        long_signals[long_condition] = 1.0
        
        # Set short signals
        short_signals[short_condition] = -1.0
        
        # Combine signals
        signals = pd.Series(long_signals + short_signals, index=df.index, dtype=np.float64)
        
        # Warmup protection
        signals.iloc[:warmup] = 0.0
        
        return signals