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
        rsi = indicators['rsi']
        bollinger = indicators['bollinger']
        atr = indicators['atr']

        for i in range(1, len(df)):
            if (df['close'][i] < bollinger['lower'][i]) & (rsi[i] < 30) & (rsi[i] > rsi[i-1]):
                signals[i] = 1.0  # LONG
            elif (df['close'][i] > bollinger['upper'][i]) & (rsi[i] > 70) & (rsi[i] < rsi[i-1]):
                signals[i] = -1.0  # SHORT
        return signals