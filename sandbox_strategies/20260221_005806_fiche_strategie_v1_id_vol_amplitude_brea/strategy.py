from utils.parameters import ParameterSpec
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Volatility Amplitude Breakout')

    @property
    def required_indicators(self) -> List[str]:
        return ['amplitude_hunter', 'donchian', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'warmup': 50, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.25,
                max_val=4.0,
                default=2.25,
                param_type='float',
                step=None,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=3.5,
                param_type='float',
                step=None,
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
         # Replace 'pd' with 'df' wherever it appears in the code