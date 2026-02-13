from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")

    @property
    def required_indicators(self):
        return ['keltner', 'supertrend', 'atr']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)
        
        # Extract indicators
        keltner_upper = indicators['keltner']['upper']
        keltner_lower = indicators['keltner']['lower']
        supertrend_value = indicators['supertrend']['supertrend']
        supertrend_direction = indicators['supertrend']['direction']
        atr = indicators['atr']
        close = df['close'].values
        open_ = df['open'].values
        
        # Create boolean masks for conditions
        long_condition = (
            (close > keltner_upper) & 
            (supertrend_direction > 0) & 
            (close > open_) & 
            (atr > np.roll(atr, 1))
        )
        
        short_condition = (
            (close < keltner_lower) & 
            (supertrend_direction < 0) & 
            (close < open_) & 
            (atr > np.roll(atr, 1))
        )
        
        # Generate signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0
        
        return signals