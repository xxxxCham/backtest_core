from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_ema_rsi_strategy_v2")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "ema", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, 
                "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(type=int, min=50, max=90),
            "rsi_oversold": ParameterSpec(type=int, min=10, max=40),
            "rsi_period": ParameterSpec(type=int, min=2, max=20),
            "stop_atr_mult": ParameterSpec(type=float, min=1.0, max=2.5),
            "tp_atr_mult": ParameterSpec(type=float, min=2.0, max=4.0),
            "warmup": ParameterSpec(type=int, min=20, max=100)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Get indicators with NaN handling
        bollinger = indicators["bollinger"]
        ema_9 = np.nan_to_num(indicators["ema"]["9"])
        ema_21 = np.nan_to_num(indicators["ema"]["21"])
        ema_50 = np.nan_to_num(indicators["ema"]["50"])
        rsi = np.nan_to_num(indicators["rsi"])
        close = np.nan_to_num(df["close"])
        
        # Warmup period
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Entry conditions
        for i in range(warmup, len(df)):
            # LONG entries
            if (
                (close[i] > ema_21[i]) & (rsi[i] > params["rsi_overbought"]) or
                (close[i] > np.nan_to_num(bollinger["upper"])[i]) & (rsi[i] > params["rsi_overbought"] - 10)
            ):
                signals.iloc[i] = 1.0
                
            # SHORT entries
            elif (
                (close[i] < ema_21[i]) & (rsi[i] < params["rsi_oversold"]) or
                (close[i] < np.nan_to_num(bollinger["lower"])[i]) & (rsi[i] < params["rsi_oversold"] + 10)
            ):
                signals.iloc[i] = -1.0
                
            # Exit conditions
            if signals.iloc[i] != 0.0:
                # Check for opposite band exit
                if (
                    (signals.iloc[i] == 1.0 and close[i] < np.nan_to_num(bollinger["lower"])[i]) or
                    (signals.iloc[i] == -1.0 and close[i] > np.nan_to_num(bollinger["upper"])[i])
                ):
                    signals.iloc[i] = 0.0
                    
                # Check for RSI divergence
                elif (
                    (signals.iloc[i] == 1.0 and rsi[i] < rsi[i-1]) or
                    (signals.iloc[i] == -1.0 and rsi[i] > rsi[i-1])
                ):
                    signals.iloc[i] = 0.0
        
        return signals