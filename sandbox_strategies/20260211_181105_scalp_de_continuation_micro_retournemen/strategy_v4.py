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
                type="int",
                min=50,
                max=90,
                default=70,
                description="RSI overbought level"
            ),
            "rsi_oversold": ParameterSpec(
                type="int",
                min=10,
                max=50,
                default=30,
                description="RSI oversold level"
            ),
            "rsi_period": ParameterSpec(
                type="int",
                min=2,
                max=50,
                default=14,
                description="RSI period"
            ),
            "stop_atr_mult": ParameterSpec(
                type="float",
                min=1.0,
                max=3.0,
                default=1.5,
                description="Stop loss multiplier based on ATR"
            ),
            "tp_atr_mult": ParameterSpec(
                type="float",
                min=2.0,
                max=5.0,
                default=3.0,
                description="Take profit multiplier based on ATR"
            ),
            "warmup": ParameterSpec(
                type="int",
                min=20,
                max=100,
                default=50,
                description="Warmup period to ignore initial signals"
            )
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Warmup period
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Get indicators with NaN protection
        bollinger = np.nan_to_num(indicators["bollinger"])
        rsi = np.nan_to_num(indicators["rsi"])
        ema = np.nan_to_num(indicators["ema"])
        
        for i in range(warmup, len(df)):
            # Current price conditions
            upper_band = bollinger[i, 0]
            lower_band = bollinger[i, 1]
            close = df.iloc[i].close
            
            # EMA condition
            ema_21 = ema[i, 1]  # Assuming ema[period=21] is second element
            
            # RSI conditions
            rsi_ob = params["rsi_overbought"]
            rsi_os = params["rsi_oversold"]
            
            # Check for long entry
            if (close > upper_band) and (rsi[i] < rsi_os) and (close > ema_21):
                signals.iloc[i] = 1.0  # LONG
                
            # Check for short entry  
            elif (close < lower_band) and (rsi[i] > rsi_ob) and (close < ema_21):
                signals.iloc[i] = -1.0  # SHORT
                
            # Exit conditions
            elif (close < upper_band and signals.iloc[i-1] == 1.0) or \
                 (close > lower_band and signals.iloc[i-1] == -1.0):
                signals.iloc[i] = 0.0  # FLAT
                
            # Stop loss conditions based on Bollinger reversal
            elif ((signals.iloc[i-1] == 1.0 and close < lower_band) or 
                  (signals.iloc[i-1] == -1.0 and close > upper_band)):
                signals.iloc[i] = 0.0  # FLAT
                
            # RSI divergence exit
            elif (rsi[i] < rsi_ob and signals.iloc[i-1] == -1.0) or \
                 (rsi[i] > rsi_os and signals.iloc[i-1] == 1.0):
                signals.iloc[i] = 0.0  # FLAT
                
        return signals