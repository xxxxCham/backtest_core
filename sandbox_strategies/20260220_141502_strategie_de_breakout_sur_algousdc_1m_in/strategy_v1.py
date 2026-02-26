from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Breakout Based on EMA20, OBV and ATR')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'obv', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_threshold': 20,
         'ema_values_needed': 2,
         'leverage': 1,
         'obv_value': 0,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'obv_value': ParameterSpec(
                name='obv_value',
                min_val=-100,
                max_val=100,
                default=0,
                param_type='float',
                step=0.1,
            ),
            'ema_values_needed': ParameterSpec(
                name='ema_values_needed',
                min_val=2,
                max_val=50,
                default=2,
                param_type='int',
                step=1,
            ),
            'adx_threshold': ParameterSpec(
                name='adx_threshold',
                min_val=10,
                max_val=100,
                default=20,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
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
        ema = np.array(indicators['ema'])
        obv = np.array(indicators['obv'])
        atr = np.array(indicators['atr'])


        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Implement your logic here to compute signals based on EMA20, OBV and ATR
        signals.iloc[:warmup] = 0.0
        return signals
