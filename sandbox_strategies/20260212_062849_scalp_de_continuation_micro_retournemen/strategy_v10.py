from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")
    
    @property
    def required_indicators(self):
        return ['ema', 'rsi', 'bollinger', 'atr']
    
    @property
    def default_params(self):
        return {
            'ema_short': 9,
            'ema_long': 21,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'bollinger_std': 2,
            'atr_multiplier': 1.5
        }
    
    def generate_signals(self, df, indicators, params):
        # Extract parameters
        ema_short = params['ema_short']
        ema_long = params['ema_long']
        rsi_oversold = params['rsi_oversold']
        rsi_overbought = params['rsi_overbought']
        bollinger_std = params['bollinger_std']
        atr_multiplier = params['atr_multiplier']
        
        signals = pd.Series(0.0, index=df.index)
        
        # EMA arrays
        ema9 = indicators["ema"]
        ema21 = indicators["ema"]
        
        # RSI array
        rsi = indicators["rsi"]
        
        # Bollinger Bands
        bb_upper = indicators['bollinger']['upper']
        bb_middle = indicators['bollinger']['middle']
        bb_lower = indicators['bollinger']['lower']
        
        # Calculate band expansion
        prev_upper = np.roll(bb_upper, 1)
        prev_lower = np.roll(bb_lower, 1)
        band_expanding = (bb_upper > prev_upper) & (bb_lower < prev_lower)
        
        # Support and resistance identification
        is_support = (df['close'] >= ema21) & (df['close'] <= df['high'].shift(1))
        is_resistance = (df['close'] <= ema21) & (df['close'] >= df['low'].shift(1))
        
        # Price near support (within bottom 10% of recent range)
        recent_range = df[['high', 'low']].diff(1).max(axis=1)
        price_near_support = (df['close'] - bb_lower) < 0.1 * recent_range.values
        
        # Price near resistance (within top 10% of recent range)
        price_near_resistance = (bb_upper - df['close']) < 0.1 * recent_range.values
        
        # Long conditions
        condition1 = (df['close'] < ema21) & (df['close'] > ema9)
        condition2 = (rsi > rsi_oversold) & is_support
        long_condition_A = condition1 & condition2
        
        condition3 = band_expanding & price_near_support
        long_condition_B = condition3
        
        long_signals = long_condition_A | long_condition_B
        signals.loc[long_signals] = 1.0
        
        # Short conditions
        condition1 = (df['close'] > ema21) & (df['close'] < ema9)
        condition2 = (rsi < rsi_oversold) & is_resistance
        short_condition_A = condition1 & condition2
        
        condition3 = band_expanding & price_near_resistance
        short_condition_B = condition3
        
        short_signals = short_condition_A | short_condition_B
        signals.loc[short_signals] = -1.0
        
        return signals