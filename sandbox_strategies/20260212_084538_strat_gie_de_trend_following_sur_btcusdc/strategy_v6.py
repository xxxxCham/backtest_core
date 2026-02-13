from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")

    @property
    def required_indicators(self):
        return ['sma', 'adx', 'atr']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)
        
        # Extract indicators
        sma_50 = indicators["sma"]
        adx = indicators['adx']['adx']
        atr = indicators['atr']
        close = df['close'].values
        
        # Define threshold
        adx_threshold = 25.0
        
        # Create boolean masks for long condition
        long_condition = (close > sma_50) & (adx > adx_threshold)
        
        # Generate signals
        signals[long_condition] = 1.0
        
        return signals