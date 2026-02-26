from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ema_macd_scalp')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'macd', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'atr_period': 14,
            'ema_period': 20,
            'leverage': 1,
            'macd_fast': 12,
            'macd_signal': 9,
            'macd_slow': 26,
            'stop_atr_mult': 1.0,
            'tp_atr_mult': 2.0,
            'warmup': 30,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=5,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'macd_fast': ParameterSpec(
                name='macd_fast',
                min_val=5,
                max_val=20,
                default=12,
                param_type='int',
                step=1,
            ),
            'macd_slow': ParameterSpec(
                name='macd_slow',
                min_val=20,
                max_val=50,
                default=26,
                param_type='int',
                step=1,
            ),
            'macd_signal': ParameterSpec(
                name='macd_signal',
                min_val=5,
                max_val=20,
                default=9,
                param_type='float',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=20,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=10,
                max_val=50,
                default=30,
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
        warmup = int(params.get('warmup', 30))

        # Ensure indicator arrays are numpy arrays
        ema_arr = np.asarray(indicators['ema'])
        macd_arr = np.asarray(indicators['macd']['macd'])
        signal_arr = np.asarray(indicators['macd']['signal'])
        hist_arr = np.asarray(indicators['macd']['histogram'])

        # Long and short conditions with proper parenthesis to avoid precedence issues
        long_cond = (
            (df['close'].values > ema_arr)
            & (macd_arr > signal_arr)
            & (hist_arr > 0)
        )
        short_cond = (
            (df['close'].values < ema_arr)
            & (macd_arr < signal_arr)
            & (hist_arr < 0)
        )

        signals[long_cond] = 1.0
        signals[short_cond] = -1.0

        # Zero out warmup period
        signals.iloc[:warmup] = 0.0
        return signals