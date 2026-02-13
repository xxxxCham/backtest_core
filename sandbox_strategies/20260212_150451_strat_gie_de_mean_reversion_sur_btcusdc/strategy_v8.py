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
        return ['donchian', 'rsi', 'atr']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)
        
        # Extract indicators
        donchian_lower = indicators['donchian']['lower']
        donchian_middle = indicators['donchian']['middle']
        rsi = indicators['rsi']
        atr = indicators['atr']
        
        # Define parameters
        rsi_oversold = 30
        
        # Create boolean masks for long condition
        close = df['close'].values
        condition1 = close <= donchian_lower
        condition2 = rsi < rsi_oversold
        condition3 = close > donchian_middle
        
        # Combine conditions with bitwise AND
        long_condition = condition1 & condition2 & condition3
        
        # Generate signals
        signals[long_condition] = 1.0
        
        return signals