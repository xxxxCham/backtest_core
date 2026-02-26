from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='stoch_supertrend_adx_atr_v2')

    @property
    def required_indicators(self) -> List[str]:
        return ['stochastic', 'supertrend', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'adx_period': 14,
            'leverage': 1,
            'stochastic_d_period': 3,
            'stochastic_k_period': 14,
            'stochastic_smooth_k': 3,
            'stop_atr_mult': 1.1,
            'supertrend_atr_period': 10,
            'supertrend_multiplier': 3.0,
            'tp_atr_mult': 2.7,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stochastic_k_period': ParameterSpec(
                name='stochastic_k_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stochastic_d_period': ParameterSpec(
                name='stochastic_d_period',
                min_val=1,
                max_val=10,
                default=3,
                param_type='int',
                step=1,
            ),
            'stochastic_smooth_k': ParameterSpec(
                name='stochastic_smooth_k',
                min_val=1,
                max_val=10,
                default=3,
                param_type='int',
                step=1,
            ),
            'supertrend_atr_period': ParameterSpec(
                name='supertrend_atr_period',
                min_val=5,
                max_val=20,
                default=10,
                param_type='int',
                step=1,
            ),
            'supertrend_multiplier': ParameterSpec(
                name='supertrend_multiplier',
                min_val=1.0,
                max_val=5.0,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
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
                default=1.1,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=2.7,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=10,
                max_val=200,
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
        n = len(df)
        warmup = int(params.get('warmup', 50))

        # initialise output series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # extract indicator arrays
        stoch = indicators['stochastic']
        k = np.nan_to_num(stoch['stoch_k']).astype(float)

        supertrend = indicators['supertrend']
        direction = np.nan_to_num(indicators['supertrend']["direction"]).astype(float)

        adx_dict = indicators['adx']
        adx_val = np.nan_to_num(adx_dict['adx']).astype(float)

        atr = np.nan_to_num(indicators['atr']).astype(float)
        close = df['close'].values.astype(float)

        # --- entry logic ---
        entry_long = (k < 20) & (direction == 1) & (adx_val > 25)
        entry_short = (k > 80) & (direction == -1) & (adx_val > 25)

        # --- exit logic ---
        prev_k = np.roll(k, 1)
        prev_k[0] = np.nan
        cross_up_80 = (k > 80) & (prev_k <= 80)
        cross_down_20 = (k < 20) & (prev_k >= 20)

        # supertrend direction flip (ignore first bar)
        prev_dir = np.roll(direction, 1)
        super_flip = (direction != prev_dir) & (np.arange(n) > 0)

        exit_long = cross_up_80 | super_flip
        exit_short = cross_down_20 | super_flip

        # combine masks
        long_mask = entry_long & ~exit_long
        short_mask = entry_short & ~exit_short

        # apply warmup protection
        long_mask[:warmup] = False
        short_mask[:warmup] = False

        # assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # --- risk management (ATR based SL/TP) ---
        df.loc[:, 'bb_stop_long'] = np.nan
        df.loc[:, 'bb_tp_long'] = np.nan
        df.loc[:, 'bb_stop_short'] = np.nan
        df.loc[:, 'bb_tp_short'] = np.nan

        stop_mult = float(params.get('stop_atr_mult', 1.1))
        tp_mult = float(params.get('tp_atr_mult', 2.7))

        if long_mask.any():
            df.loc[long_mask, 'bb_stop_long'] = close[long_mask] - stop_mult * atr[long_mask]
            df.loc[long_mask, 'bb_tp_long'] = close[long_mask] + tp_mult * atr[long_mask]

        if short_mask.any():
            df.loc[short_mask, 'bb_stop_short'] = close[short_mask] + stop_mult * atr[short_mask]
            df.loc[short_mask, 'bb_tp_short'] = close[short_mask] - tp_mult * atr[short_mask]

        return signals