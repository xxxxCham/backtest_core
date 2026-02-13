from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")

    @property
    def required_indicators(self):
        return ['rsi', 'bollinger', 'atr', 'ema']

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
        ema_50 = indicators['ema']
        
        # Define thresholds
        rsi_oversold = 30
        
        # Create boolean masks for long condition
        cond1 = df['close'] <= bb_lower
        cond2 = rsi < rsi_oversold
        cond3 = rsi < np.roll(rsi, 1)
        cond4 = df['close'] > ema_50
        
        # Combine conditions with bitwise AND
        long_condition = cond1 & cond2 & cond3 & cond4
        
        # Generate signals
        signals[long_condition] = 1.0
        
        return signals