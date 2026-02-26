from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_macd_atr_divergence_filter')

    @property
    def required_indicators(self) -> List[str]:
        return ['momentum', 'macd', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'leverage': 1,
         'macd_fast': 12,
         'macd_signal': 9,
         'macd_slow': 26,
         'momentum_period': 10,
         'stop_atr_mult': 1.4,
         'tp_atr_mult': 2.1,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'momentum_period': ParameterSpec(
                name='momentum_period',
                min_val=5,
                max_val=20,
                default=10,
                param_type='int',
                step=1,
            ),
            'macd_fast': ParameterSpec(
                name='macd_fast',
                min_val=6,
                max_val=20,
                default=12,
                param_type='int',
                step=1,
            ),
            'macd_slow': ParameterSpec(
                name='macd_slow',
                min_val=20,
                max_val=40,
                default=26,
                param_type='int',
                step=1,
            ),
            'macd_signal': ParameterSpec(
                name='macd_signal',
                min_val=5,
                max_val=15,
                default=9,
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
                default=1.4,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.1,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        # Compute previous values for comparison
        momentum_prev = np.roll(indicators['momentum'], 1)
        close_prev = np.roll(df['close'].values, 1)

        # MACD components
        macd_values = indicators['macd']['macd']
        indicators['macd']['signal'] = indicators['macd']['signal']

        # ATR mean for threshold
        atr_mean = np.mean(indicators['atr'])

        # Long and short conditions
        long_cond = (
            (macd_values > indicators['macd']['signal']) &
            (indicators['momentum'] > momentum_prev) &
            (df['close'].values > close_prev) &
            (indicators['atr'] > atr_mean)
        )

        short_cond = (
            (macd_values < indicators['macd']['signal']) &
            (indicators['momentum'] < momentum_prev) &
            (df['close'].values < close_prev) &
            (indicators['atr'] > atr_mean)
        )

        # Assign signals
        signals[long_cond] = 1.0
        signals[short_cond] = -1.0
        signals.iloc[:warmup] = 0.0
        return signals
