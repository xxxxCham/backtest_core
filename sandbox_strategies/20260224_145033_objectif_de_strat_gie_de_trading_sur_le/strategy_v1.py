from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Phase Lock Proposal')

    @property
    def required_indicators(self) -> List[str]:
        return ['adx', 'ema', 'ichimoku']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'ema_length': 20,
         'gating_volatility': 'implicit',
         'ichimoku_span': 'cloud',
         'leverage': 1,
         'mode_inverse': True,
         'sp_mult': 0.5,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'tp_mult': 2.8,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=1,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'ema_length': ParameterSpec(
                name='ema_length',
                min_val=2,
                max_val=100,
                default=20,
                param_type='int',
                step=1,
            ),
            'tp_mult': ParameterSpec(
                name='tp_mult',
                min_val=1.3,
                max_val=5.0,
                default=2.8,
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
        # Implement logic for generating buy and sell signals here. Use long_mask 
        # and short_mask arrays to track long positions and shorts respectively.
        signals.iloc[:warmup] = 0.0
        return signals
