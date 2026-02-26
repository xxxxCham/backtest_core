from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Phase Lock')

    @property
    def required_indicators(self) -> List[str]:
        return ['adx']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ADX_period': 14,
         'EMA_long_period': 26,
         'EMA_short_period': 16,
         'fees': 0.003,
         'leverage': 1,
         'slippage': 0.005,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ADX_period': ParameterSpec(
                name='ADX_period',
                min_val=10,
                max_val=200,
                default=14,
                param_type='int',
                step=1,
            ),
            'EMA_short_period': ParameterSpec(
                name='EMA_short_period',
                min_val=8,
                max_val=30,
                default=16,
                param_type='int',
                step=1,
            ),
            'EMA_long_period': ParameterSpec(
                name='EMA_long_period',
                min_val=20,
                max_val=50,
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

            # implement Phase Lock logic here
        signals.iloc[:warmup] = 0.0
        return signals
