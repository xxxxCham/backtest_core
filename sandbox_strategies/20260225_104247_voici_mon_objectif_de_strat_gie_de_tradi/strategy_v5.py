from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='snake_case_name')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'aroon', 'macd']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'aroon_length': 5,
         'atr_period': 14,
         'bollinger_deviation': 2.0,
         'bollinger_period': 20,
         'ema_long': 26,
         'ema_short': 12,
         'leverage': 1,
         'macd_fast_length': 12,
         'macd_slow_length': 26,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_short': ParameterSpec(
                name='ema_short',
                min_val=8,
                max_val=50,
                default=12,
                param_type='int',
                step=1,
            ),
            'ema_long': ParameterSpec(
                name='ema_long',
                min_val=14,
                max_val=60,
                default=26,
                param_type='int',
                step=1,
            ),
            'aroon_length': ParameterSpec(
                name='aroon_length',
                min_val=5,
                max_val=99,
                default=5,
                param_type='int',
                step=1,
            ),
            'macd_fast_length': ParameterSpec(
                name='macd_fast_length',
                min_val=8,
                max_val=30,
                default=12,
                param_type='int',
                step=1,
            ),
            'macd_slow_length': ParameterSpec(
                name='macd_slow_length',
                min_val=17,
                max_val=60,
                default=26,
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
                default=1.5,
                param_type='float',
                step=0.1,
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
        def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)
            n = len(df)

            # Your logic to generate signals should go here. This is a placeholder for now.

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
