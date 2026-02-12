from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")

    @property
    def required_indicators(self):
        return ['rsi', 'bollinger', 'atr']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index)
        
        # Extract necessary data as numpy arrays
        close = df['close'].values
        
        # Get bollinger bands and RSI from indicators
        bollinger = indicators['bollinger']
        upper_band = bollinger['upper']
        lower_band = bollinger['lower']
        rsi = indicators['rsi']
        
        # Calculate long and short conditions using vectorized operations
        long_mask = (close > upper_band) & (rsi < 30)
        short_mask = (close < lower_band) & (rsi > 70)
        
        # Assign signals based on conditions
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        
        return signals