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
        return ['bollinger', 'vwap', 'atr', 'rsi']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)
        
        # Extract indicator arrays
        close = df['close'].values
        bb_upper = indicators['bollinger']['upper']
        bb_middle = indicators['bollinger']['middle']
        bb_lower = indicators['bollinger']['lower']
        vwap = indicators['vwap']
        rsi = indicators['rsi']
        
        # Create boolean masks for long condition
        long_condition_1 = close <= bb_lower
        long_condition_2 = vwap <= close
        long_condition_3 = rsi < 30
        
        # Combine long conditions with bitwise AND
        long_mask = long_condition_1 & long_condition_2 & long_condition_3
        
        # Create boolean masks for short condition
        short_condition_1 = close >= bb_upper
        short_condition_2 = vwap >= close
        short_condition_3 = rsi > 70
        
        # Combine short conditions with bitwise AND
        short_mask = short_condition_1 & short_condition_2 & short_condition_3
        
        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        
        return signals