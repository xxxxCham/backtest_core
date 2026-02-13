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
        return ['macd', 'rsi', 'atr']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        macd = indicators['macd']['macd']
        signal = indicators['macd']['signal']
        rsi = indicators['rsi']
        long_mask = (np.roll(macd, 1) < np.roll(signal, 1)) & (macd > signal) & (rsi < 30)
        
        short_mask = (np.roll(macd, 1) > np.roll(signal, 1)) & (macd < signal) & (rsi > 70)
        
        signals  = pd.Series(0.0, index=df.index)
        signals[long_mask] = 1.0  # Long when the MACD line crosses above its signal line and RSI is below oversold level (30).
        signals[short_mask] = -1.0  # Short when the MACD line crosses below its signal line and RSI is above overbought level (70).
        
        return signals