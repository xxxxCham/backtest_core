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
        
        # Extract indicators
        sma = indicators['sma']
        aroon = indicators['aroon']
        atr = indicators['atr']
        
        # Calculate sma slope (difference between current and previous sma values)
        sma_slope = np.zeros_like(sma)
        sma_slope[1:] = sma[1:] - sma[:-1]
        sma_slope[0] = 0
        
        # Extract aroon values
        aroon_up = aroon['aroon_up']
        aroon_down = aroon['aroon_down']
        
        # Generate signals
        # LONG when: (sma_slope > 0) & (aroon_up > aroon_down)
        long_condition = (sma_slope > 0) & (aroon_up > aroon_down)
        
        # Convert boolean mask to signal values (1.0 for long, 0.0 for no signal)
        signals[long_condition] = 1.0
        
        return signals