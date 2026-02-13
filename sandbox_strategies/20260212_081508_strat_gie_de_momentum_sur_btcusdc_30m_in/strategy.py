from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")

    @property
    def required_indicators(self):
        return ['momentum', 'macd', 'atr']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)
        
        # Extract indicator arrays
        momentum_arr = indicators["momentum"]
        macd_signal = indicators['macd']['signal']
        macd_histogram = indicators['macd']['histogram']
        
        # Create conditions
        long_conditions = (
            (momentum_arr > 0) &
            (macd_signal > 0) &
            (macd_histogram > 0)
        )
        
        short_conditions = (
            (momentum_arr < 0) &
            (macd_signal < 0) &
            (macd_histogram < 0)
        )
        
        # Apply signals
        signals[long_conditions] = 1.0
        signals[short_conditions] = -1.0
        
        return signals