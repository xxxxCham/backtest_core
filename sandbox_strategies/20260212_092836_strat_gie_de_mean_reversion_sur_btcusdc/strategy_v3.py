from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")

    @property
    def required_indicators(self):
        return ['rsi', 'bollinger', 'atr', 'ema']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)
        
        # Extract indicator arrays
        rsi = indicators['rsi']
        bb_upper = indicators['bollinger']['upper']
        bb_middle = indicators['bollinger']['middle']
        bb_lower = indicators['bollinger']['lower']
        ema_50 = indicators['ema']
        close = df['close'].values
        rsi_overbought = 70.0
        
        # Create boolean masks for conditions
        short_condition = (close >= bb_upper) & (rsi > rsi_overbought) & (close < ema_50)
        
        # Generate signals
        signals[short_condition] = -1.0
        
        return signals