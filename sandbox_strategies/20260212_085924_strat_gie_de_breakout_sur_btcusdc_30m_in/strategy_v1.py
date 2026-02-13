from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")

    @property
    def required_indicators(self):
        return ['bollinger', 'supertrend', 'atr']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)
        
        # Extract indicators
        close = df['close'].values
        volume = df['volume'].values
        bollinger_upper = indicators['bollinger']['upper']
        bollinger_middle = indicators['bollinger']['middle']
        supertrend_values = indicators['supertrend']['supertrend']
        supertrend_direction = indicators['supertrend']['direction']
        atr_values = indicators['atr']
        
        # Calculate 20-period average volume
        avg_volume = np.convolve(volume, np.ones(20)/20, mode='valid')
        avg_volume = np.concatenate([np.full(len(volume) - len(avg_volume), np.nan), avg_volume])
        
        # Create boolean masks for conditions
        # Close crosses above Upper Bollinger Band (current close > upper band and previous close <= upper band)
        close_above_upper = (close > bollinger_upper) & (np.roll(close, 1) <= np.roll(bollinger_upper, 1))
        
        # Supertrend is Up (supertrend direction = 1)
        is_supertrend_up = supertrend_direction == 1
        
        # Volume is above its 20-period average
        volume_above_avg = volume > avg_volume
        
        # Combine all conditions for LONG signal
        long_condition = close_above_upper & is_supertrend_up & volume_above_avg
        
        # Convert boolean mask to signals
        signals[long_condition] = 1.0
        
        return signals