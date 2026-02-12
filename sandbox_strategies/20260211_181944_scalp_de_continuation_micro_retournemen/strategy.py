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
        return ["rsi", "bollinger", "ema"]

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
            "rsi_overbought": ParameterSpec(float, (0, 100)),
            "rsi_oversold": ParameterSpec(float, (0, 100)),
            "rsi_period": ParameterSpec(int, (1, 100)),
            "stop_atr_mult": ParameterSpec(float, (0.1, 5)),
            "tp_atr_mult": ParameterSpec(float, (0.1, 5)),
            "warmup": ParameterSpec(int, (10, 100))
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
        
        # Get indicators
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        lower_bb = np.nan_to_num(bb["lower"])
        upper_bb = np.nan_to_num(bb["upper"])
        ema_trend = np.nan_to_num(indicators["ema"])
        
        # Get ATR for stop and profit levels
        atr = np.nan_to_num(indicators["atr"])
        
        for i in range(warmup, len(df)):
            current_rsi = rsi[i]
            close_price = df.loc[i, "close"]
            
            # Determine trend direction
            ema_up = ema_trend[i] > 0
            ema_down = ema_trend[i] < 0
            
            # Long entry conditions
            if current_rsi > params["rsi_oversold"] and close_price < lower_bb[i]:
                if ema_up:
                    signals.iloc[i] = 1.0
                    
                    # Take profit at upper BB or next support level
                    if close_price >= upper_bb[i]:
                        signals.iloc[i] = 0.0
                    
                    # Stop loss below recent low
                    sl_level = close_price - atr[i] * params["stop_atr_mult"]
                    if close_price <= sl_level:
                        signals.iloc[i] = 0.0
                        
            # Short entry conditions
            elif current_rsi < params["rsi_overbought"] and close_price > upper_bb[i]:
                if ema_down:
                    signals.iloc[i] = -1.0
                    
                    # Take profit at lower BB or next resistance level
                    if close_price <= lower_bb[i]:
                        signals.iloc[i] = 0.0
                    
                    # Stop loss above recent high
                    sl_level = close_price + atr[i] * params["stop_atr_mult"]
                    if close_price >= sl_level:
                        signals.iloc[i] = 0.0
        
        return signals