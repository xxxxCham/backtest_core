from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='MomentumADARSIROCATDXStopLossTakeProfit')

    @property
    def required_indicators(self) -> List[str]:
        return ['momentum', 'roc', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_threshold': 1.5,
         'leverage': 1,
         'momentum_period': 14,
         'roc_threshold': 25,
         'stop_atr_mult': 1.5,
         'stop_loss_multiplier': 2.3,
         'take_profit_multiplier': 4,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'momentum_period': ParameterSpec(
                name='momentum_period',
                min_val=8,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'roc_threshold': ParameterSpec(
                name='roc_threshold',
                min_val=-25,
                max_val=75,
                default=25,
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
        def generate_signals(self, df):
            # Your logic here for generating signals based on momentum, roc and atr

            return self._generate_signals()  # Replace this line with your actual implementation.
        signals.iloc[:warmup] = 0.0
        return signals
