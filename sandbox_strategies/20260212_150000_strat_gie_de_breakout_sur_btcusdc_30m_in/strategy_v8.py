from utils.parameters import ParameterSpec
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")

    @property
    def required_indicators(self):
        return ['keltner', 'supertrend', 'atr', 'rsi']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)
        
        # Extract indicator arrays
        close = df['close'].values
        keltner_upper = indicators['keltner']['upper']
        keltner_lower = indicators['keltner']['lower']
        supertrend_line = indicators['supertrend']['supertrend']
        rsi = indicators['rsi']
        
        # Create boolean masks for long conditions
        cond1 = close > keltner_upper
        cond2 = supertrend_line < close
        cond3 = rsi < 30
        
        # Long signal when all conditions are true
        long_mask = cond1 & cond2 & cond3
        
        # Create boolean masks for short conditions
        cond4 = close < keltner_lower
        cond5 = supertrend_line > close
        cond6 = rsi > 70
        
        # Short signal when all conditions are true
        short_mask = cond4 & cond5 & cond6
        
        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        
        return signals