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
        return ['vortex', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'aroon_downtrend_limit': 65,
         'aroon_uptrend_limit': 35,
         'atr_multiplier': 1.5,
         'atr_period': 14,
         'leverage': 2,
         'rsi_threshold': 70,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_threshold': ParameterSpec(
                name='rsi_threshold',
                min_val=30,
                max_val=90,
                default=70,
                param_type='int',
                step=1,
            ),
            'aroon_uptrend_limit': ParameterSpec(
                name='aroon_uptrend_limit',
                min_val=0,
                max_val=100,
                default=35,
                param_type='int',
                step=1,
            ),
            'aroon_downtrend_limit': ParameterSpec(
                name='aroon_downtrend_limit',
                min_val=0,
                max_val=100,
                default=65,
                param_type='int',
                step=1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=2,
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

            # implement your logic here to generate signals based on vortex, rsi and atr
            long_mask = np.zeros(n, dtype=bool)
            short_mask = np.zeros(n, dtype=bool)

            # warmup protection
            warmup = int(params.get("warmup", 50))
            signals.iloc[:warmup] = 0.0

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
