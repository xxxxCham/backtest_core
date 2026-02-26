from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ema_adx_trend_1h')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'adx_period': 14,
            'atr_period': 14,
            'ema_long_period': 200,
            'ema_short_period': 50,
            'leverage': 1,
            'stop_atr_mult': 1.5,
            'tp_atr_mult': 3.0,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_short_period': ParameterSpec(
                name='ema_short_period',
                min_val=10,
                max_val=100,
                default=50,
                param_type='int',
                step=1,
            ),
            'ema_long_period': ParameterSpec(
                name='ema_long_period',
                min_val=20,
                max_val=200,
                default=200,
                param_type='int',
                step=1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=10,
                max_val=50,
                default=14,
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
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=3.0,
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

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        """
        Generate long (+1) and short (-1) signals based on EMA crossover and ADX filter.
        """
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        warmup = int(params.get('warmup', 50))

        # Retrieve EMA arrays from the indicators dictionary.
        # Expected structure: indicators['ema'] contains 'short' and 'long' arrays.
        ema_short = indicators['ema']
        ema_long = indicators['ema']

        # Cross above and below masks using np.roll to compare with previous bar.
        cross_above = (ema_short > ema_long) & (np.roll(ema_short, 1) <= np.roll(ema_long, 1))
        cross_above[0] = False
        cross_below = (ema_short < ema_long) & (np.roll(ema_short, 1) >= np.roll(ema_long, 1))
        cross_below[0] = False

        # ADX mask: filter for strong trend
        adx_mask = indicators['adx']['adx'] > 25

        # Entry conditions
        close_arr = df['close'].values
        long_cond = (close_arr > ema_short) & cross_above & adx_mask
        short_cond = (close_arr < ema_short) & cross_below & adx_mask

        # Assign signals
        signals[long_cond] = 1.0
        signals[short_cond] = -1.0

        # Zero out warmup period
        signals.iloc[:warmup] = 0.0
        return signals