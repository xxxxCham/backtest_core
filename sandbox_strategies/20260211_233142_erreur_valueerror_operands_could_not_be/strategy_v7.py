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
        close = df['close'].values

        # Get RSI values
        rsi = indicators['rsi']
        rsi_oversold = 30  # Example value; adjust as needed
        rsi_overbought = 70  # Example value; adjust as needed

        # Get Bollinger Bands values
        bollinger_upper = indicators['bollinger']['upper']
        bollinger_lower = indicators['bollinger']['lower']

        # Create LONG condition: close > upper_bollinger & rsi < rsi_oversold
        long_condition = (close > bollinger_upper) & (rsi < rsi_oversold)
        signals.loc[long_condition] = 1.0

        # Create SHORT condition: close < lower_bollinger & rsi > rsi_overbought
        short_condition = (close < bollinger_lower) & (rsi > rsi_overbought)
        signals.loc[short_condition] = -1.0

        return signals