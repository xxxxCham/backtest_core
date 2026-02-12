from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")

    @property
    def required_indicators(self):
        return ['rsi', 'bollinger', 'atr']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)
        
        # Extract indicator values
        rsi = indicators['rsi']
        bollinger = indicators['bollinger']
        upper_band = bollinger['upper']
        lower_band = bollinger['lower']
        middle_band = bollinger['middle']
        price = df['close'].values
        
        # Create long signals
        long_condition = (price < lower_band) & (rsi < 30)
        signals[long_condition] = 1.0
        
        # Create short signals
        short_condition = (price > upper_band) & (rsi > 70)
        signals[short_condition] = -1.0
        
        return signals