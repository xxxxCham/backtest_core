from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='phase_lock_v2')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr', 'ema']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'ema_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 20}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=5.0,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=0,
                max_val=100,
                default=20,
                param_type='int',
                step=1,
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
        
        ema_values = indicators['ema']
        atr_values = indicators['atr']
        
        # Calculate rolling mean for ATR confirmation
        atr_mean = pd.Series(atr_values).rolling(window=20).mean().values
        atr_mean[0:20] = np.nan  # Keep NaN for warmup period
        
        # Entry long conditions: close > EMA AND ATR > 1.5 * ATR_mean AND price > EMA
        entry_long = (df['close'] > ema_values) & (atr_values > 1.5 * atr_mean) & (df['close'] > ema_values)
        
        # Entry short conditions: close < EMA AND ATR > 1.5 * ATR_mean AND price < EMA
        entry_short = (df['close'] < ema_values) & (atr_values > 1.5 * atr_mean) & (df['close'] < ema_values)
        
        # Combine masks
        long_mask = entry_long
        short_mask = entry_short
        
        # Apply warmup masking
        long_mask[0:warmup] = False
        short_mask[0:warmup] = False
        
        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        
        return signals