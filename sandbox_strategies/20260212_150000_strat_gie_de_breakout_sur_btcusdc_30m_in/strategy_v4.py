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
        return ['keltner', 'supertrend', 'atr', 'rsi']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)
        
        # Extract indicator values
        keltner_upper = indicators['keltner']['upper']
        keltner_lower = indicators['keltner']['lower']
        supertrend_direction = indicators['supertrend']['direction']
        rsi = indicators['rsi']
        close = df['close'].values
        
        # Set default RSI thresholds
        rsi_oversold = 30
        rsi_overbought = 70
        
        # Create boolean masks for long and short conditions
        long_condition = (
            (close > keltner_upper) & 
            (supertrend_direction > 0) & 
            (rsi > rsi_oversold)
        )
        
        short_condition = (
            (close < keltner_lower) & 
            (supertrend_direction < 0) & 
            (rsi < rsi_overbought)
        )
        
        # Generate signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0
        
        return signals