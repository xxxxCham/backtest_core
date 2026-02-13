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
        return ['rsi', 'bollinger', 'atr']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)
        
        # Extract indicators
        rsi = indicators['rsi']
        bb_upper = indicators['bollinger']['upper']
        bb_middle = indicators['bollinger']['middle']
        bb_lower = indicators['bollinger']['lower']
        atr = indicators['atr']
        
        # Define thresholds
        rsi_overbought = 70
        rsi_oversold = 30
        
        # Generate signals
        # LONG when: (close >= bb_upper * 0.99) & (rsi > rsi_overbought)
        close = df['close'].values
        long_condition = (close >= bb_upper * 0.99) & (rsi > rsi_overbought)
        
        # SHORT when: False
        short_condition = np.full(len(df), False, dtype=bool)
        
        # Apply signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0
        
        return signals