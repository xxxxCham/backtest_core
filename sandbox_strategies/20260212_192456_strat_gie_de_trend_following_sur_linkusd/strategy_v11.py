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
        return ['supertrend', 'adx', 'atr']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)
        
        # Extract indicator values
        supertrend_direction = indicators['supertrend']['direction']
        adx_value = indicators['adx']['adx']
        atr_value = indicators['atr']
        
        # Set thresholds
        adx_threshold = 25.0
        atr_threshold = 1.0
        
        # Create boolean masks for long condition
        long_condition_1 = supertrend_direction > 0
        long_condition_2 = adx_value >= adx_threshold
        long_condition_3 = atr_value > atr_threshold
        long_condition_4 = df['close'] > indicators['supertrend']['supertrend']
        
        # Combine all conditions with bitwise AND
        long_mask = long_condition_1 & long_condition_2 & long_condition_3 & long_condition_4
        
        # Generate signals
        signals[long_mask] = 1.0
        
        return signals