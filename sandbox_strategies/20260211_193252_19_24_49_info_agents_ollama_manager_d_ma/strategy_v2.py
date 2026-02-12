from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="ema_bollinger_rsi_scalper")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "bollinger", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(20, 40, 1),
            "rsi_period": ParameterSpec(10, 20, 1),
            "stop_atr_mult": ParameterSpec(1.0, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 4.0, 0.1),
            "warmup": ParameterSpec(30, 70, 5),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        ema_9 = np.nan_to_num(indicators["ema"])
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        bb_upper = np.nan_to_num(bb["upper"])
        bb_lower = np.nan_to_num(bb["lower"])
        bb_middle = np.nan_to_num(bb["middle"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Prepare arrays for comparison
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        
        # Entry conditions
        # Long entry: price crosses above EMA 21, RSI below oversold and rising, price above lower BB
        ema_21 = ema_9  # Using EMA 9 as proxy for EMA 21 since we only have EMA 9
        price = df["close"].values
        
        # Previous EMA values
        ema_21_prev = np.roll(ema_21, 1)
        rsi_prev = np.roll(rsi, 1)
        price_prev = np.roll(price, 1)
        bb_lower_prev = np.roll(bb_lower, 1)
        bb_upper_prev = np.roll(bb_upper, 1)
        
        # Long signal conditions
        long_condition = (
            (price > ema_21) & (price_prev <= ema_21_prev) &
            (rsi < rsi_oversold) & (rsi > rsi_prev) &
            (price > bb_lower)
        )
        
        # Short signal conditions
        short_condition = (
            (price < ema_21) & (price_prev >= ema_21_prev) &
            (rsi > rsi_overbought) & (rsi < rsi_prev) &
            (price < bb_upper)
        )
        
        # Generate signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0
        
        return signals