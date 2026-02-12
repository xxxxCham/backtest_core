from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_ema_rsi_strategy")

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
            "rsi_overbought": ParameterSpec(type=float, bounds=(30, 70), default=70),
            "rsi_oversold": ParameterSpec(type=float, bounds=(30, 70), default=30),
            "rsi_period": ParameterSpec(type=int, bounds=(2, 100), default=14),
            "stop_atr_mult": ParameterSpec(type=float, bounds=(0.5, 2.0), default=1.5),
            "tp_atr_mult": ParameterSpec(type=float, bounds=(1.0, 4.0), default=3.0),
            "warmup": ParameterSpec(type=int, bounds=(20, 100), default=50)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        bollinger = indicators["bollinger"]
        bb_lower = np.nan_to_num(bollinger["lower"])
        bb_upper = np.nan_to_num(bollinger["upper"])
        bb_middle = np.nan_to_num(bollinger["middle"])
        
        ema = np.nan_to_num(indicators["ema"])
        rsi = np.nan_to_num(indicators["rsi"])
        
        for i in range(warmup, len(df)):
            current_close = df.iloc[i]["close"]
            current_rsi = rsi[i]
            current_ema = ema[i]
            lower_band = bb_lower[i]
            upper_band = bb_upper[i]
            
            # Entry conditions
            if (current_close > current_ema and current_rsi > params["rsi_overbought"]) or \
               (current_close < current_ema and current_rsi < params["rsi_oversold"]):
                if (current_close > upper_band or current_close < lower_band):
                    signals.iloc[i] = 1.0 if current_close > current_ema else -1.0
                    
            # Exit conditions
            if signals.iloc[i-1] != 0.0:
                if (current_close > upper_band or current_close < lower_band):
                    signals.iloc[i] = 0.0
                    
                # Check for RSI divergence
                prev_rsi = rsi[i-1]
                if signals.iloc[i-1] == 1.0 and (prev_rsi < current_rsi):
                    signals.iloc[i] = 0.0
                elif signals.iloc[i-1] == -1.0 and (prev_rsi > current_rsi):
                    signals.iloc[i] = 0.0
        
        return signals