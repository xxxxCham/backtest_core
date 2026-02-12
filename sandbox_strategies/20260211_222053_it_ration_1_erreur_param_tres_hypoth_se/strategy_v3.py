from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="continuation_momentum_v2")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "atr", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "momentum_multiplier": 0.5,
            "momentum_period": 10,
            "ema_long_period": 21,
            "ema_short_period": 12,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 1.2,
            "tp_atr_mult": 2.5,
            "volatility_squeeze_threshold": 0.1,
            "warmup": 30
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "momentum_multiplier": ParameterSpec(type_=(float, int), default=0.5, min_val=0.1, max_val=5.0),
            "momentum_period": ParameterSpec(type_=int, default=10, min_val=1, max_val=100),
            "ema_long_period": ParameterSpec(type_=int, default=21, min_val=2, max_val=100),
            "ema_short_period": ParameterSpec(type_=int, default=12, min_val=1, max_val=20),
            "rsi_overbought": ParameterSpec(type_=(int, float), default=70, min_val=1, max_val=100),
            "rsi_oversold": ParameterSpec(type_=(int, float), default=30, min_val=0, max_val=99),
            "rsi_period": ParameterSpec(type_=int, default=14, min_val=1, max_val=100),
            "stop_atr_mult": ParameterSpec(type_=(float, int), default=1.2, min_val=0.1, max_val=10.0),
            "tp_atr_mult": ParameterSpec(type_=(float, int), default=2.5, min_val=0.1, max_val=10.0),
            "volatility_squeeze_threshold": ParameterSpec(type_=(float, int), default=0.1, min_val=0.01, max_val=1.0),
            "warmup": ParameterSpec(type_=int, default=30, min_val=0, max_val=1000)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract parameters
        momentum_multiplier = params["momentum_multiplier"]
        momentum_period = params["momentum_period"]
        ema_long_period = params["ema_long_period"]
        ema_short_period = params["ema_short_period"]
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        rsi_period = params["rsi_period"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        volatility_squeeze_threshold = params["volatility_squeeze_threshold"]
        warmup = int(params["warmup"])
        
        # Calculate momentum
        price = np.nan_to_num(df["close"].values)
        momentum = np.zeros_like(price)
        if momentum_period > 1:
            momentum[1:] = price[1:] - price[:-1]
            momentum = momentum / price[:-1]
        
        # Extract indicator arrays and handle nans
        ema_long = np.nan_to_num(indicators["ema"][f"ema_{ema_long_period}"])
        ema_short = np.nan_to_num(indicators["ema"][f"ema_{ema_short_period}"])
        atr = np.nan_to_num(indicators["atr"])
        rsi_value = np.nan_to_num(indicators["rsi"][f"rsi_{rsi_period}"])
        
        # Price relative to EMAs
        above_ema_long = (price > ema_long)
        below_ema_short = (price < ema_short)
        above_ema_short = (price > ema_short)
        below_ema_long = (price < ema_long)
        
        # Momentum conditions
        momentum_long = momentum > momentum_multiplier
        momentum_short = momentum < -momentum_multiplier
        
        # RSI conditions
        rsi_long = (rsi_value < rsi_overbought)
        rsi_short = (rsi_value < rsi_oversold)
        
        # Volatility squeeze - calculate compressed range
        high = np.nan_to_num(df["high"].values)
        low = np.nan_to_num(df["low"].values)
        compressed_range = high - low
        expanded_range = (high - low) / (high + low + 1e-10)
        squeeze = expanded_range < volatility_squeeze_threshold
        
        # Warmback protection
        signals.iloc[:warmup] = 0.0
        
        # Long conditions
        long_conditions = (
            above_ema_long & 
            below_ema_short &
            momentum_long &
            rsi_long &
            squeeze
        )
        
        # Short conditions
        short_conditions = (
            below_ema_long &
            above_ema_short &
            momentum_short &
            rsi_short &
            squeeze
        )
        
        # Apply conditions
        signals[long_conditions] = 1.0
        signals[short_conditions] = -1.0
        
        return signals