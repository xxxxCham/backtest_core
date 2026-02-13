from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="momentum_avaxusdc_1d")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "macd", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(param_type="int", min_value=60, max_value=80, step=5),
            "rsi_oversold": ParameterSpec(param_type="int", min_value=10, max_value=30, step=5),
            "rsi_period": ParameterSpec(param_type="int", min_value=10, max_value=20, step=2),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=2.0, step=0.2),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=4.0, step=0.5),
            "warmup": ParameterSpec(param_type="int", min_value=30, max_value=70, step=10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        rsi = np.nan_to_num(indicators["rsi"])
        macd_histogram = np.nan_to_num(indicators["macd"]["histogram"])
        atr = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Entry conditions
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_period = params.get("rsi_period", 14)
        
        # Entry long: RSI crosses above 50 and MACD histogram is positive and crossing above zero
        rsi_crossed_above_50 = (rsi[1:] > 50) & (rsi[:-1] <= 50)
        macd_positive = macd_histogram > 0
        macd_crossing_above_zero = (macd_histogram[1:] > 0) & (macd_histogram[:-1] <= 0)
        long_entry = rsi_crossed_above_50 & macd_positive[1:] & macd_crossing_above_zero
        
        # Entry short: RSI crosses below 50 and MACD histogram is negative and crossing below zero
        rsi_crossed_below_50 = (rsi[1:] < 50) & (rsi[:-1] >= 50)
        macd_negative = macd_histogram < 0
        macd_crossing_below_zero = (macd_histogram[1:] < 0) & (macd_histogram[:-1] >= 0)
        short_entry = rsi_crossed_below_50 & macd_negative[1:] & macd_crossing_below_zero
        
        # Exit conditions
        # RSI crosses below 30 (long exit) or crosses above 70 (short exit)
        rsi_crossed_below_30 = (rsi[1:] < rsi_oversold) & (rsi[:-1] >= rsi_oversold)
        rsi_crossed_above_70 = (rsi[1:] > rsi_overbought) & (rsi[:-1] <= rsi_overbought)
        
        # Price diverges from RSI (simple version: price makes higher highs while RSI makes lower highs)
        # This is a simplified version for demonstration
        price_highs = np.array([close[i] for i in range(1, len(close)) if i > 0 and close[i] > close[i-1]])
        rsi_highs = np.array([rsi[i] for i in range(1, len(rsi)) if i > 0 and rsi[i] > rsi[i-1]])
        divergent_highs = (len(price_highs) > 0 and len(rsi_highs) > 0 and 
                           price_highs[-1] > price_highs[-2] and rsi_highs[-1] < rsi_highs[-2])
        
        # Long exits
        long_exit = rsi_crossed_below_30 | divergent_highs
        
        # Short exits
        short_exit = rsi_crossed_above_70 | divergent_highs
        
        # Generate signals
        entries = np.zeros_like(rsi, dtype=bool)
        exits = np.zeros_like(rsi, dtype=bool)
        
        # Long entries
        entries[1:][long_entry] = True
        # Short entries
        entries[1:][short_entry] = True
        
        # Long exits
        exits[1:][long_exit] = True
        # Short exits
        exits[1:][short_exit] = True
        
        # Initialize signal array
        signal_values = np.zeros_like(rsi)
        
        # Set entry signals
        signal_values[entries] = 1.0  # Long
        signal_values[entries] = -1.0  # Short (this is a placeholder, actual logic needs to distinguish long/short)
        
        # Set exit signals
        signal_values[exits] = 0.0  # Flat
        
        # Simplified version for now - better to track positions
        for i in range(len(rsi)):
            if i > 0 and entries[i]:
                signal_values[i] = 1.0 if rsi[i] > 50 else -1.0
            elif exits[i]:
                signal_values[i] = 0.0
                
        signals.iloc[1:] = signal_values[1:]
        
        # Set warmup to zero
        signals.iloc[:warmup] = 0.0
        
        return signals