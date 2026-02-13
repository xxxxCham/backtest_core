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
        return ['bollinger', 'stoch_rsi', 'atr']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)
        
        # Extract indicators
        close = df['close'].values
        bb_lower = indicators['bollinger']['lower']
        stoch_rsi_k = indicators['stoch_rsi']['k']
        atr = indicators["atr"]
        
        # Create boolean masks for long condition
        long_condition = (close < bb_lower) & (stoch_rsi_k < 20)
        
        # Generate signals
        signals[long_condition] = 1.0
        
        return signals