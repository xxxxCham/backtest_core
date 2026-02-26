from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='trend_supertrend_ema_adx')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'adx', 'ema']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'ema_fast_period': 20,
         'ema_slow_period': 50,
         'leverage': 1,
         'stop_atr_mult': 2.0,
         'supertrend_atr_period': 16,
         'supertrend_multiplier': 3.5,
         'tp_atr_mult': 4.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'supertrend_atr_period': ParameterSpec(
                name='supertrend_atr_period',
                min_val=10,
                max_val=30,
                default=16,
                param_type='int',
                step=1,
            ),
            'supertrend_multiplier': ParameterSpec(
                name='supertrend_multiplier',
                min_val=1.5,
                max_val=5.0,
                default=3.5,
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
            'ema_fast_period': ParameterSpec(
                name='ema_fast_period',
                min_val=5,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'ema_slow_period': ParameterSpec(
                name='ema_slow_period',
                min_val=10,
                max_val=100,
                default=50,
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
                max_val=6.0,
                default=4.0,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        # Extract indicator arrays
        supertrend_dir = indicators['supertrend']['direction']
        adx_val = indicators['adx']['adx']
        ema_fast = indicators['ema']
        ema_slow = indicators['ema']

        # Long and short signal masks
        long_mask = (supertrend_dir == 1) & (adx_val > 30) & (ema_fast > ema_slow)
        short_mask = (supertrend_dir == -1) & (adx_val > 30) & (ema_fast < ema_slow)

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals.iloc[:warmup] = 0.0
        return signals
