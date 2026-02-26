from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='regime_adaptatif_xplusdc_30m')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr', 'adx', 'sma']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'adx_period': 14,
            'atr_period': 14,
            'atr_threshold': 0.5,
            'leverage': 1,
            'sma_period': 20,
            'stop_atr_mult': 1.4,
            'tp_atr_mult': 2.4,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'sma_period': ParameterSpec(
                name='sma_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'atr_threshold': ParameterSpec(
                name='atr_threshold',
                min_val=0.1,
                max_val=2.0,
                default=0.5,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.4,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.4,
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
        # Ensure warmup does not exceed array length
        warmup = min(warmup, n)

        # Extract indicator arrays
        close = df['close'].values
        sma = indicators['sma']
        atr = indicators['atr']
        adx = indicators['adx']['adx']

        # Parameter values
        atr_threshold = params.get('atr_threshold', 0.5)

        # Long and short masks
        long_mask = (close > sma) & ((atr > atr_threshold) | ((atr <= atr_threshold) & (adx < 25)))
        short_mask = (close < sma) & ((atr > atr_threshold) | ((atr <= atr_threshold) & (adx < 25)))

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Zero out warmup period
        signals[:warmup] = 0.0

        return signals