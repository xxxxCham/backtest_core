from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")

    @property
    def required_indicators(self):
        return ['rsi', 'bollinger', 'atr', 'adx']

    @property
    def default_params(self):
        return {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'adx_threshold': 20
        }

    def generate_signals(self, df, indicators, params):
        rsi_oversold = params.get('rsi_oversold', 30)
        rsi_overbought = params.get('rsi_overbought', 70)
        adx_threshold = params.get('adx_threshold', 20)
        
        signals = pd.Series(0.0, index=df.index)
        
        # Extract indicators
        rsi = indicators['rsi']
        bollinger = indicators['bollinger']
        upper = bollinger['upper']
        middle = bollinger['middle']
        lower = bollinger['lower']
        adx = indicators['adx']
        price = df['close'].values
        
        # Create masks for long and short conditions
        long_condition = (price < lower) & (rsi < rsi_oversold) & (adx > adx_threshold)
        short_condition = (price > upper) & (rsi > rsi_overbought) & (adx > adx_threshold)
        
        # Generate signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0
        
        return signals