from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")
    
    @property
    def required_indicators(self):
        return ['bollinger', 'ema', 'atr']
    
    @property
    def default_params(self):
        return {
            'bollinger_period': 20,
            'bollinger_std': 2.0,
            'ema_fast': 12,
            'ema_slow': 21,
            'atr_period': 14
        }
    
    def generate_signals(self, df, indicators, params):
        # Extract parameters
        bollinger_period = params['bollinger_period']
        bollinger_std = params['bollinger_std']
        ema_fast = params['ema_fast']
        ema_slow = params['ema_slow']
        atr_period = params['atr_period']
        
        # Get indicator arrays
        bollinger = indicators['bollinger']
        ema = indicators['ema']
        atr = indicators['atr']
        
        # Bollinger Bands
        upper_band = bollinger['upper']
        middle_band = bollinger['middle']
        lower_band = bollinger['lower']
        
        # EMA values
        ema_fast_arr = ema['ema_{}'.format(ema_fast)]
        ema_slow_arr = ema['ema_{}'.format(ema_slow)]
        
        # Long conditions
        cond1_long = df['close'] > upper_band
        cond2_long = ema_fast_arr > ema_slow_arr
        cond3_long = self._check_expanding(atr, atr_period)
        
        long_signals = cond1_long & cond2_long & cond3_long
        
        # Short conditions
        cond1_short = df['close'] < lower_band
        cond2_short = ema_fast_arr < ema_slow_arr
        cond3_short = self._check_expanding(atr, atr_period)
        
        short_signals = cond1_short & cond2_short & cond3_short
        
        # Create signals
        signals = pd.Series(0.0, index=df.index)
        signals[long_signals] = 1.0
        signals[short_signals] = -1.0
        
        return signals
    
    def _check_expanding(self, atr_arr: np.ndarray, period: int) -> np.ndarray:
        """
        Check if ATR is expanding (increasing)
        """
        # Calculate ATR differences from previous period
        atr_diff = np.diff(atr_arr, periods=period)
        
        # Pad with False at the beginning (no previous value for first period)
        is_expanding = np.zeros(len(atr_arr), dtype=bool)
        is_expanding[period:] = (atr_diff > 0)
        
        return is_expanding