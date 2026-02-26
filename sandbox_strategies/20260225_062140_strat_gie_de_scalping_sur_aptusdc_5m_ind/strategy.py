from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='aptusdc_5m_ema_bollinger_atr_scalp_v3')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'atr_period': 14,
            'atr_threshold': 0.001,
            'bollinger_period': 20,
            'bollinger_std_dev': 2.0,
            'ema_period': 9,
            'leverage': 1,
            'stop_atr_mult': 2.3,
            'tp_atr_mult': 4.6,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=5,
                max_val=20,
                default=9,
                param_type='int',
                step=1,
            ),
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'bollinger_std_dev': ParameterSpec(
                name='bollinger_std_dev',
                min_val=1.5,
                max_val=3.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=7,
                max_val=21,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_threshold': ParameterSpec(
                name='atr_threshold',
                min_val=0.0005,
                max_val=0.005,
                default=0.001,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.3,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=8.0,
                default=4.6,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=20,
                max_val=100,
                default=50,
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

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        n = len(df)
        warmup = int(params.get('warmup', 50))

        # Prepare output series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Extract needed data
        close_prices = df['close'].values
        ema = indicators['ema']                     # numpy array
        boll_upper = indicators['bollinger']['upper']
        boll_lower = indicators['bollinger']['lower']
        atr = indicators['atr']
        atr_threshold = params.get('atr_threshold', 0.001)

        # Boolean masks for entry conditions
        long_mask = (
            (close_prices > ema) &
            (close_prices > boll_upper) &
            (atr > atr_threshold)
        )
        short_mask = (
            (close_prices < ema) &
            (close_prices < boll_lower) &
            (atr > atr_threshold)
        )

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Zero out warmup period
        signals.iloc[:warmup] = 0.0

        return signals