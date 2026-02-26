from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Fibonacci_Levels_EMA_OBV')

    @property
    def required_indicators(self) -> List[str]:
        return ['fibonacci_levels', 'ema', 'obv']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'bollinger_stddev': 2,
         'buffer': 0.02,
         'fib_level1': 0.85,
         'fib_level2': 0.9,
         'fib_level3': 0.8,
         'fib_level4': 0.75,
         'fib_level5': 0.6,
         'fib_level6': 0.5,
         'fib_level7': 0.2,
         'fib_level8': 0.1,
         'leverage': 1,
         'obv_window': 14,
         'stop_atr_mult': 1.33,
         'tp_atr_mult': 1.67,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'fib_level1': ParameterSpec(
                name='fib_level1',
                min_val=0.8,
                max_val=1,
                default=0.85,
                param_type='float',
                step=0.1,
            ),
            'fib_level2': ParameterSpec(
                name='fib_level2',
                min_val=0.9,
                max_val=1,
                default=0.9,
                param_type='float',
                step=0.1,
            ),
            'fib_level3': ParameterSpec(
                name='fib_level3',
                min_val=0.7,
                max_val=1,
                default=0.85,
                param_type='float',
                step=0.1,
            ),
            'fib_level4': ParameterSpec(
                name='fib_level4',
                min_val=0.6,
                max_val=1,
                default=0.9,
                param_type='float',
                step=0.1,
            ),
            'fib_level5': ParameterSpec(
                name='fib_level5',
                min_val=0.3,
                max_val=0.8,
                default=0.7,
                param_type='float',
                step=0.1,
            ),
            'fib_level6': ParameterSpec(
                name='fib_level6',
                min_val=0.1,
                max_val=0.2,
                default=0.4,
                param_type='float',
                step=0.1,
            ),
            'fib_level7': ParameterSpec(
                name='fib_level7',
                min_val=0.5,
                max_val=0.8,
                default=0.3,
                param_type='float',
                step=0.1,
            ),
            'fib_level8': ParameterSpec(
                name='fib_level8',
                min_val=1,
                max_val=2,
                default=0.9,
                param_type='float',
                step=0.1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=5,
                max_val=60,
                default=14,
                param_type='int',
                step=1,
            ),
            'bollinger_stddev': ParameterSpec(
                name='bollinger_stddev',
                min_val=1,
                max_val=2,
                default=2,
                param_type='float',
                step=0.1,
            ),
            'obv_window': ParameterSpec(
                name='obv_window',
                min_val=5,
                max_val=60,
                default=14,
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
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=2.0,
                default=1.33,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=1.67,
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
        # implement explicit LONG / SHORT / FLAT logic
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)


        # Write SL/TP columns into df if using ATR-based risk management
        signals.iloc[:warmup] = 0.0
        return signals
