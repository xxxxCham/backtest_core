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
        return ['bollinger', 'vwap', 'atr']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)
        
        # Extract indicator arrays
        bb_upper = indicators['bollinger']['upper']
        bb_middle = indicators['bollinger']['middle']
        bb_lower = indicators['bollinger']['lower']
        vwap = indicators['vwap']
        close = df['close'].values
        atr = indicators['atr']
        
        # Create boolean masks for long and short conditions
        long_condition = (close < bb_lower) & (vwap < close)
        short_condition = (close > bb_upper) & (vwap > close)
        
        # Generate signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0
        
        return signals