from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")

    @property
    def required_indicators(self):
        return ['bollinger', 'ema', 'rsi']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)
        
        # Get indicator data
        bollinger = indicators['bollinger']
        ema_21 = indicators['ema']
        rsi = indicators['rsi']
        
        # LONG condition: close below lower band AND above EMA 21 AND RSI >50
        long_mask = (
            (df.close < bollinger['lower']) &
            (df.close > ema_21) &
            (rsi > 50)
        )
        signals[long_mask] = 1.0
        
        # SHORT condition: close above upper band AND below EMA 21 AND RSI <50
        short_mask = (
            (df.close > bollinger['upper']) &
            (df.close < ema_21) &
            (rsi < 50)
        )
        signals[short_mask] = -1.0
        
        return signals