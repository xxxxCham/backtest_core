from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='aptusdc_5m_ema_bollinger_atr_scalp_v2')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'atr_period': 14,
            'atr_threshold': 0.001,
            'bollinger_period': 20,
            'bollinger_std_dev': 2,
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
                default=2,
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

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)

        warmup = int(params.get('warmup', 50))
        atr_threshold = float(params.get('atr_threshold', 0.001))

        # Extract raw arrays
        close = df['close'].values
        ema_arr = indicators['ema']                     # plain array
        bb_upper = indicators['bollinger']['upper']    # dict sub‑key
        bb_lower = indicators['bollinger']['lower']
        atr_arr = indicators['atr']

        # Previous values for breakout detection
        prev_close = np.roll(close, 1)
        prev_bb_upper = np.roll(bb_upper, 1)
        prev_bb_lower = np.roll(bb_lower, 1)

        # Long and short entry conditions
        long_cond = (
            (close > ema_arr) &
            (close > bb_upper) &
            (prev_close <= prev_bb_upper) &
            (atr_arr > atr_threshold)
        )
        short_cond = (
            (close < ema_arr) &
            (close < bb_lower) &
            (prev_close >= prev_bb_lower) &
            (atr_arr > atr_threshold)
        )

        signals[long_cond] = 1.0
        signals[short_cond] = -1.0

        # Zero out warm‑up period
        signals.iloc[:warmup] = 0.0
        return signals