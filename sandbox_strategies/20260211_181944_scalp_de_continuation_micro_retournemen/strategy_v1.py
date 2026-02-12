from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="ema_rsi_bollinger")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi", "bollinger"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "bollinger_period": 20,
            "ema_periods": [9, 21, 50],
            "rsi_period": 14,
            "std_dev": 2,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "warmup": 50
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "bollinger_period": ParameterSpec(int, min=5, max=50),
            "ema_periods": ParameterSpec(list, subtype=int, min=5, max=100),
            "rsi_period": ParameterSpec(int, min=2, max=50),
            "std_dev": ParameterSpec(float, min=1.0, max=3.0),
            "stop_atr_mult": ParameterSpec(float, min=1.0, max=2.0),
            "tp_atr_mult": ParameterSpec(float, min=2.0, max=4.0),
            "warmup": ParameterSpec(int, min=10, max=100)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Get indicators
        ema_9 = np.nan_to_num(indicators["ema"][0])
        ema_21 = np.nan_to_num(indicators["ema"][1])
        ema_50 = np.nan_to_num(indicators["ema"][2])
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        bb_upper = np.nan_to_num(bb["upper"])
        bb_lower = np.nan_to_num(bb["lower"])
        
        # Warmup period
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Initialize previous values for exit conditions
        prev_close = df.close.shift(1)
        prev_bb_upper = bb_upper.shift(1)
        prev_bb_lower = bb_lower.shift(1)
        
        for i in range(warmup, len(df)):
            current_close = df.close.iloc[i]
            current_rsi = rsi[i]
            current_bb_upper = bb_upper[i]
            current_bb_lower = bb_lower[i]
            
            # Entry conditions
            if (current_close > ema_21[i] and 
                current_rsi > 50 and 
                current_close < current_bb_upper):
                signals.iloc[i] = 1.0
                
            elif (current_close < ema_21[i] and 
                  current_rsi < 50 and 
                  current_close > current_bb_lower):
                signals.iloc[i] = -1.0
                
            else:
                # Exit conditions
                if ((current_close > prev_bb_upper and signals.iloc[i-1] == 1.0) or
                    (current_close < prev_bb_lower and signals.iloc[i-1] == -1.0)):
                    signals.iloc[i] = 0.0
                    
                elif (rsi[i] < rsi[i-1] and signals.iloc[i-1] == 1.0 and current_rsi > 30):
                    signals.iloc[i] = 0.0
                    
                elif (rsi[i] > rsi[i-1] and signals.iloc[i-1] == -1.0 and current_rsi < 70):
                    signals.iloc[i] = 0.0
                    
                else:
                    signals.iloc[i] = signals.iloc[i-1]
        
        return signals