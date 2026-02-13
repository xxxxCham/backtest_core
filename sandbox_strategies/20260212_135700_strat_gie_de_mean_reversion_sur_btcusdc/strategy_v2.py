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
        return ['keltner', 'cci', 'atr']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)
        
        # Extract indicator arrays
        keltner_upper = indicators['keltner']['upper']
        cci_values = indicators["cci"]
        close = df['close'].values
        
        # Create boolean masks for long condition
        long_condition_1 = close > keltner_upper
        long_condition_2 = cci_values < -100
        long_condition_3 = cci_values > np.roll(cci_values, 1)
        
        # Combine all conditions with bitwise AND
        long_mask = long_condition_1 & long_condition_2 & long_condition_3
        
        # Generate signals
        signals[long_mask] = 1.0
        
        return signals