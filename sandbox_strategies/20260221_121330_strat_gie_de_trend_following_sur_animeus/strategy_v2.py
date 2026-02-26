from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Vortex Ichimoku Trend Following on ANIMEUSDC 30m')

    @property
    def required_indicators(self) -> List[str]:
        return ['vortex', 'ichimoku', 'sma', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'conversion_line_width': 3,
         'fees': 10.0,
         'ichimoku_span': 26,
         'leverage': 2,
         'slippage': 5.0,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'vortex_period': 9,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'vortex_period': ParameterSpec(
                name='vortex_period',
                min_val=7,
                max_val=40,
                default=9,
                param_type='int',
                step=1,
            ),
            'ichimoku_span': ParameterSpec(
                name='ichimoku_span',
                min_val=12,
                max_val=80,
                default=26,
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

            # implement your logic here to compute the signals based on df and parameters
        signals.iloc[:warmup] = 0.0
        return signals
