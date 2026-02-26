from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='adx_bollinger_reversion')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'atr_period': 14,
         'bollinger_period': 20,
         'bollinger_std_dev': 2,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 2.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=10,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=1,
                param_type='int',
                step=1,
            ),
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        
        # Get indicator arrays
        bollinger_upper = indicators['bollinger']['upper']
        bollinger_lower = indicators['bollinger']['lower']
        bollinger_middle = indicators['bollinger']['middle']
        adx_adx = indicators['adx']['adx']
        atr_array = indicators['atr']
        
        # Create shifted arrays for comparison
        close_prev = df['close'].shift(1).fillna(method='bfill')
        bollinger_lower_prev = np.roll(bollinger_lower, 1)
        bollinger_lower_prev[0] = bollinger_lower_prev[1]  # Handle first element
        bollinger_upper_prev = np.roll(bollinger_upper, 1)
        bollinger_upper_prev[0] = bollinger_upper_prev[1]  # Handle first element
        
        # Cross below lower band: current close <= lower AND previous close > previous lower
        cross_below_lower = (df['close'].values <= bollinger_lower) & (close_prev.values > bollinger_lower_prev)
        
        # Cross above upper band: current close >= upper AND previous close < previous upper
        cross_above_upper = (df['close'].values >= bollinger_upper) & (close_prev.values < bollinger_upper_prev)
        
        # ADX condition
        adx_weak = adx_adx < 25
        
        # Entry conditions
        long_condition = cross_below_lower & adx_weak
        short_condition = cross_above_upper & adx_weak
        
        # Exit conditions: cross middle band OR strong ADX
        cross_above_middle = (df['close'].values >= bollinger_middle) & (close_prev.values < bollinger_middle)
        cross_below_middle = (df['close'].values <= bollinger_middle) & (close_prev.values > bollinger_middle)
        adx_strong = adx_adx > 40
        exit_condition = cross_above_middle | cross_below_middle | adx_strong
        
        # Apply signals
        signals.loc[long_condition] = 1.0
        signals.loc[short_condition] = -1.0
        signals.loc[exit_condition] = 0.0
        
        signals.iloc[:warmup] = 0.0
        return signals