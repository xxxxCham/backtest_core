from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_ema")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "rsi", "ema"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "warmup": 50
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(
                type=int,
                default=70,
                min=50,
                max=90,
                description="RSI overbought level"
            ),
            "rsi_oversold": ParameterSpec(
                type=int,
                default=30,
                min=10,
                max=70,
                description="RSI oversold level"
            ),
            "rsi_period": ParameterSpec(
                type=int,
                default=14,
                min=5,
                max=20,
                description="RSI period"
            ),
            "stop_atr_mult": ParameterSpec(
                type=float,
                default=1.5,
                min=1.0,
                max=2.0,
                description="Stop loss multiple of ATR"
            ),
            "tp_atr_mult": ParameterSpec(
                type=float,
                default=3.0,
                min=2.0,
                max=4.0,
                description="Take profit multiple of ATR"
            ),
            "warmup": ParameterSpec(
                type=int,
                default=50,
                min=20,
                max=100,
                description="Warmup period to filter initial signals"
            )
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Warmup period to avoid early signals
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Get indicators
        bollinger = indicators["bollinger"]
        rsi = np.nan_to_num(indicators["rsi"])
        ema = indicators["ema"]
        
        # Entry conditions
        long_entry = (
            df.close > np.nan_to_num(bollinger["upper"]) &
            rsi < params["rsi_oversold"] &
            ema["slow"] > ema["very_slow"]
        )
        
        short_entry = (
            df.close < np.nan_to_num(bollinger["lower"]) &
            rsi > params["rsi_overbought"] &
            ema["slow"] < ema["very_slow"]
        )
        
        # Exit conditions based on Bollinger bands
        long_exit = df.close < np.nan_to_num(bollinger["lower"])
        short_exit = df.close > np.nan_to_num(bollinger["upper"])
        
        # Update signals
        for i in range(warmup, len(df)):
            if long_entry.iloc[i]:
                signals.iloc[i] = 1.0
            elif short_entry.iloc[i]:
                signals.iloc[i] = -1.0
            
            # Exit conditions
            if signals.iloc[i] == 1.0 and (long_exit.iloc[i] or df.close.iloc[i] > np.nan_to_num(bollinger["upper"])):
                signals.iloc[i] = 0.0
            elif signals.iloc[i] == -1.0 and (short_exit.iloc[i] or df.close.iloc[i] < np.nan_to_num(bollinger["lower"])):
                signals.iloc[i] = 0.0
        
        return signals