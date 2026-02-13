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
        return ['ema', 'aroon', 'atr']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)
        
        # Extract indicators
        ema_fast = indicators["ema"]
        ema_slow = indicators["ema"]
        aroon_up = indicators['aroon']['aroon_up']
        aroon_down = indicators['aroon']['aroon_down']
        atr = indicators['atr']
        
        # Create boolean masks for long condition
        long_condition = (ema_fast > ema_slow) & (aroon_up > aroon_down)
        
        # Generate signals
        signals[long_condition] = 1.0
        
        return signals