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
        signals = pd.Series(0.0, index=df.index)
        
        # Extract indicators
        macd_line = indicators['macd']['macd']
        signal_line = indicators['macd']['signal']
        rsi = indicators['rsi']
        atr = indicators['atr']
        ema_20 = indicators["ema"]
        price = df['close'].values
        
        # Calculate EMA crossover
        ema_cross = np.zeros_like(ema_20, dtype=bool)
        ema_cross[1:] = (ema_20[1:] > ema_20[:-1]) & (price[1:] > ema_20[1:])
        
        # Long conditions
        macd_cross_up = np.zeros_like(macd_line, dtype=bool)
        macd_cross_up[1:] = (macd_line[1:] > signal_line[1:]) & (macd_line[:-1] <= signal_line[:-1])
        
        rsi_below_50 = rsi < 50
        rsi_above_oversold = rsi > 30
        
        price_above_ema = price > ema_20
        
        long_condition = macd_cross_up & rsi_below_50 & rsi_above_oversold & price_above_ema
        
        # Short conditions
        macd_cross_down = np.zeros_like(macd_line, dtype=bool)
        macd_cross_down[1:] = (macd_line[1:] < signal_line[1:]) & (macd_line[:-1] >= signal_line[:-1])
        
        rsi_above_50 = rsi > 50
        rsi_below_overbought = rsi < 70
        
        price_below_ema = price < ema_20
        
        short_condition = macd_cross_down & rsi_above_50 & rsi_below_overbought & price_below_ema
        
        # Generate signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0
        
        return signals