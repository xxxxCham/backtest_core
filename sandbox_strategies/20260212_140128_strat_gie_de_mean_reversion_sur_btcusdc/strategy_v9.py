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
        donchian_lower = indicators['donchian']['lower']
        williams_r = indicators['williams_r']
        atr = indicators['atr']
        
        # Create boolean masks for long condition
        long_condition = (df['close'] == donchian_lower) & (williams_r < -80)
        
        # Generate signals
        signals[long_condition] = 1.0
        
        return signals