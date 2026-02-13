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
        return ['sma', 'aroon', 'atr']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)
        
        # Extract indicator values
        sma = indicators["sma"]
        aroon_up = indicators['aroon']['aroon_up']
        aroon_down = indicators['aroon']['aroon_down']
        atr = indicators['atr']
        
        # Define periods
        sma_period = 20
        aroon_period = 14
        
        # Create boolean masks for long condition
        price_above_sma = df['close'] > sma
        aroon_up_above_down = aroon_up > aroon_down
        
        # Combine conditions with bitwise AND
        long_condition = price_above_sma & aroon_up_above_down
        
        # Generate signals
        signals[long_condition] = 1.0
        
        return signals