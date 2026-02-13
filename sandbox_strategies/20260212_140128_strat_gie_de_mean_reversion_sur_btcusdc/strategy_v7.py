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
        return ['donchian', 'williams_r', 'atr']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)
        
        # Extract indicators
        donchian_upper = indicators['donchian']['upper']
        williams_r = indicators["williams_r"]
        atr = indicators["atr"]
        
        # Create boolean masks for long condition
        long_condition1 = df['close'] >= donchian_upper
        long_condition2 = williams_r < -80
        
        # Combine conditions with bitwise AND
        long_mask = long_condition1 & long_condition2
        
        # Generate signals
        signals[long_mask] = 1.0
        
        return signals