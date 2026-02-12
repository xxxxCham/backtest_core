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

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)
        
        # Extract indicator arrays
        rsi = indicators['rsi']
        upper_band = indicators['bollinger']['upper']
        middle_band = indicators['bollinger']['middle']
        lower_band = indicators['bollinger']['lower']
        close = df['close'].values
        
        # Create boolean masks for long and short conditions
        long_condition = (rsi < 30) & (close < lower_band)
        short_condition = (rsi > 70) & (close > upper_band)
        
        # Generate signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0
        
        return signals