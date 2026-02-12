from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")

    @property
    def required_indicators(self):
        return ['bollinger', 'rsi', 'ema']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)
        
        # LONG conditions
        long_boll = df['close'] > df['bollinger'].upper
        long_rsi = df['rsi'] < 30
        long_ema = df['ema_fast'] > df['ema_slow']
        signals.loc[long_boll & long_rsi & long_ema] = 1.0
        
        # SHORT conditions
        short_boll = df['close'] < df['bollinger'].lower
        short_rsi = df['rsi'] > 70
        short_ema = df['ema_fast'] < df['ema_slow']
        signals.loc[short_boll & short_rsi & short_ema] = -1.0
        
        return signals