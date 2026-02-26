from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='stoch_supertrend_atr_v4')

    @property
    def required_indicators(self) -> List[str]:
        return ['stochastic', 'supertrend', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'atr_period': 14,
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

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        n = len(df)
        warmup = int(params.get('warmup', 50))

        # initialise signal series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        signals.iloc[:warmup] = 0.0

        # ----- indicator extraction -----
        stoch = indicators['stochastic']
        k = stoch['stoch_k'].astype(float)                     # stochastic %K

        supertrend = indicators['supertrend']
        direction = indicators['supertrend']["direction"].astype(float)     # ensure float dtype

        atr = indicators['atr'].astype(float)                 # ATR values

        close = df['close'].values

        # ----- parameters -----
        stop_atr_mult = float(params.get('stop_atr_mult', 1.1))
        tp_atr_mult = float(params.get('tp_atr_mult', 2.7))

        # ----- entry conditions -----
        long_mask = (k < 20) & (direction == 1)
        short_mask = (k > 80) & (direction == -1)

        # ----- exit conditions -----
        prev_k = np.roll(k, 1).astype(float)
        prev_k[0] = np.nan
        cross_up_50 = (k > 50) & (prev_k <= 50)
        cross_down_50 = (k < 50) & (prev_k >= 50)
        cross_k_50 = cross_up_50 | cross_down_50

        prev_dir = np.roll(direction, 1).astype(float)
        prev_dir[0] = np.nan
        direction_change = direction != prev_dir

        exit_mask = cross_k_50 | direction_change

        # ----- apply signals -----
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0  # flatten positions on exit

        # ----- risk management columns -----
        df['bb_stop_long'] = np.nan
        df['bb_tp_long'] = np.nan
        df['bb_stop_short'] = np.nan
        df['bb_tp_short'] = np.nan

        if long_mask.any():
            entry_price_long = close[long_mask]
            atr_long = atr[long_mask]
            df.loc[long_mask, 'bb_stop_long'] = entry_price_long - stop_atr_mult * atr_long
            df.loc[long_mask, 'bb_tp_long'] = entry_price_long + tp_atr_mult * atr_long

        if short_mask.any():
            entry_price_short = close[short_mask]
            atr_short = atr[short_mask]
            df.loc[short_mask, 'bb_stop_short'] = entry_price_short + stop_atr_mult * atr_short
            df.loc[short_mask, 'bb_tp_short'] = entry_price_short - tp_atr_mult * atr_short

        return signals