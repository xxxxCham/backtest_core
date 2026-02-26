from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='RAYUSDC-MomentumV2')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'obv', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'obv_neglimit': -0.02,
         'obv_period': 14,
         'stop_atr_mult': 2.5,
         'tp_atr_mult': 3.0,
         'warmup': 70}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'obv_period': ParameterSpec(
                name='obv_period',
                min_val=14,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=2.0,
                max_val=6.0,
                default=3.0,
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
        def generate_signals(self, df):
            # Your logic to compute signals goes here
            # For example, you can use Bollinger bands and moving averages to generate buy/sell signals.

            close = np.array([float(x) for x in df["close"]])
        signals.iloc[:warmup] = 0.0
        return signals
