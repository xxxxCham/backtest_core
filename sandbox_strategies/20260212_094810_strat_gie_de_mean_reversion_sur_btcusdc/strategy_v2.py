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
        
        # Extract indicators
        rsi = indicators['rsi']
        bb_upper = indicators['bollinger']['upper']
        bb_middle = indicators['bollinger']['middle']
        bb_lower = indicators['bollinger']['lower']
        atr = indicators['atr']
        
        # Set thresholds
        rsi_oversold = 30
        atr_threshold = np.mean(atr)  # Using mean as threshold
        
        # Create boolean masks for long condition
        close = df['close'].values
        long_condition = (close < bb_lower) & (rsi < rsi_oversold) & (atr > atr_threshold)
        
        # Generate signals
        signals[long_condition] = 1.0
        
        return signals