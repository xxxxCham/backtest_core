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
        bollinger_upper = indicators['bollinger']['upper']
        bollinger_lower = indicators['bollinger']['lower']
        supertrend_line = indicators['supertrend']['supertrend']
        supertrend_direction = indicators['supertrend']['direction']
        atr = indicators['atr']
        price = df['close'].values
        volume = df['volume'].values
        
        # Calculate average volume over last 10 periods
        avg_volume = np.full_like(volume, np.nan)
        for i in range(10, len(volume)):
            avg_volume[i] = np.mean(volume[i-10:i])
        
        # Create boolean masks for conditions
        # LONG conditions
        price_crossed_above_upper = (price[1:] > bollinger_upper[1:]) & (price[:-1] <= bollinger_upper[:-1])
        supertrend_upward = supertrend_direction[1:] > 0
        volume_confirms_breakout = volume[1:] > 2 * avg_volume[:-1]
        
        long_condition = price_crossed_above_upper & supertrend_upward & volume_confirms_breakout
        
        # SHORT conditions
        price_crossed_below_lower = (price[1:] < bollinger_lower[1:]) & (price[:-1] >= bollinger_lower[:-1])
        supertrend_downward = supertrend_direction[1:] < 0
        volume_confirms_breakout_short = volume[1:] > 2 * avg_volume[:-1]
        
        short_condition = price_crossed_below_lower & supertrend_downward & volume_confirms_breakout_short
        
        # Generate signals
        signals.iloc[1:] = 0
        signals.iloc[1:][long_condition] = 1
        signals.iloc[1:][short_condition] = -1
        
        return signals