from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_atr_scalper")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 90, 1, "RSI Overbought Level"),
            "rsi_oversold": ParameterSpec(10, 40, 1, "RSI Oversold Level"),
            "rsi_period": ParameterSpec(5, 30, 1, "RSI Period"),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1, "Stop Loss ATR Multiplier"),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1, "Take Profit ATR Multiplier"),
            "warmup": ParameterSpec(20, 100, 1, "Warmup Period"),
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
        bb = indicators["bollinger"]
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])
        lower = np.nan_to_num(bb["lower"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Get parameters
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))
        
        # Initialize positions
        position = 0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        
        # Warmup
        signals.iloc[:warmup] = 0.0
        
        # Loop through data
        for i in range(warmup, len(df)):
            # Entry conditions
            price = df["close"].iloc[i]
            prev_rsi = rsi[i-1] if i > 0 else 0
            
            # Long entry: price crosses above lower Bollinger Band, RSI below 30, RSI rising
            if (price > lower[i] and 
                rsi[i] < rsi_oversold and 
                rsi[i] > prev_rsi):
                if position == 0:
                    position = 1
                    entry_price = price
                    stop_loss = entry_price - stop_atr_mult * atr[i]
                    take_profit = entry_price + tp_atr_mult * atr[i]
                    signals.iloc[i] = 1.0  # LONG signal
                    
            # Short entry: price crosses below upper Bollinger Band, RSI above 70, RSI falling
            elif (price < upper[i] and 
                  rsi[i] > rsi_overbought and 
                  rsi[i] < prev_rsi):
                if position == 0:
                    position = -1
                    entry_price = price
                    stop_loss = entry_price + stop_atr_mult * atr[i]
                    take_profit = entry_price - tp_atr_mult * atr[i]
                    signals.iloc[i] = -1.0  # SHORT signal
                    
            # Exit conditions
            elif position == 1:  # Long position
                if (price >= take_profit or 
                    price <= stop_loss or 
                    rsi[i] > rsi_overbought or 
                    rsi[i] < rsi_oversold):
                    position = 0
                    signals.iloc[i] = 0.0  # FLAT signal
                    
            elif position == -1:  # Short position
                if (price <= take_profit or 
                    price >= stop_loss or 
                    rsi[i] > rsi_overbought or 
                    rsi[i] < rsi_oversold):
                    position = 0
                    signals.iloc[i] = 0.0  # FLAT signal
                    
        return signals