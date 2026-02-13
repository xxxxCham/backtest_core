from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_stoch_rsi_mean_reversion_v2")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "stoch_rsi", "atr", "momentum"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"bollinger_period": 20, "bollinger_std_dev": 2, "momentum_period": 5, "stoch_rsi_overbought": 80, "stoch_rsi_oversold": 20, "stoch_rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {}

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Get indicators from the provided indicators dict
        bollinger = indicators["bollinger"]
        stoch_rsi = indicators["stoch_rsi"]
        atr = indicators["atr"]
        momentum = indicators["momentum"]
        
        # Extract Bollinger bands
        bollinger_upper = np.nan_to_num(bollinger["upper"])
        bollinger_lower = np.nan_to_num(bollinger["lower"])
        bollinger_middle = np.nan_to_num(bollinger["middle"])
        
        # Extract StochRSI values
        stoch_rsi_k = np.nan_to_num(stoch_rsi["k"])
        stoch_rsi_d = np.nan_to_num(stoch_rsi["d"])
        
        # Extract momentum values
        momentum_values = np.nan_to_num(momentum)
        
        # Extract ATR values
        atr_values = np.nan_to_num(atr)
        
        # Get parameters
        stoch_rsi_oversold = params.get("stoch_rsi_oversold", 20)
        stoch_rsi_overbought = params.get("stoch_rsi_overbought", 80)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        
        # Generate signals
        for i in range(len(df)):
            # Entry condition: price below lower Bollinger band, StochRSI oversold, and momentum positive
            if (df["close"].iloc[i] < bollinger_lower[i] and 
                stoch_rsi_k[i] < stoch_rsi_oversold and 
                momentum_values[i] > 0):
                signals.iloc[i] = 1.0  # Long signal
            
            # Exit condition: price above Bollinger middle band or StochRSI overbought
            elif (df["close"].iloc[i] > bollinger_middle[i] or 
                  stoch_rsi_k[i] > stoch_rsi_overbought):
                signals.iloc[i] = 0.0  # Exit signal
        
        # Apply warmup period
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals