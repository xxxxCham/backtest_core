from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_trend_filter_vortex')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'bollinger', 'vortex', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_avg_period': 20,
         'bollinger_period': 20,
         'bollinger_std': 2,
         'ema_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'vortex_period': 14,
         'warmup': 50}

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
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
                max_val=50,
                default=20,
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
            'atr_avg_period': ParameterSpec(
                name='atr_avg_period',
                min_val=10,
                max_val=50,
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
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        # Calculate EMA
        ema_val = indicators['ema']

        # Calculate Bollinger Bands
        indicators['bollinger']['middle'] = indicators['bollinger']['middle']
        indicators['bollinger']['upper'] = indicators['bollinger']['upper']
        indicators['bollinger']['lower'] = indicators['bollinger']['lower']

        # Calculate VORTEX oscillator
        vortex = indicators['vortex']['oscillator']

        # Create signal arrays
        long_signal = (df['close'] > ema_val) & (df['close'] < indicators['bollinger']['lower']) & (vortex > np.roll(vortex, 1))
        short_signal = (df['close'] < ema_val) & (df['close'] > indicators['bollinger']['upper']) & (vortex > np.roll(vortex, 1))

        # Assign signals
        signals[long_signal] = 1.0
        signals[short_signal] = -1.0
        signals.iloc[:warmup] = 0.0
        return signals