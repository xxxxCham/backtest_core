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
        close = df['close'].values
        lower_band = indicators['bollinger']['lower']
        atr = indicators['atr']
        
        # Create boolean masks for conditions
        short_condition = (close < lower_band) & (rsi < 30)
        
        # Generate signals
        signals[short_condition] = -1.0
        
        return signals