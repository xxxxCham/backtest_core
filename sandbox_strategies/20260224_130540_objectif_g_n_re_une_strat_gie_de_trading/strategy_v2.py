from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Counter Confirmation Contrarian with Filtered Market Signals')

    @property
    def required_indicators(self) -> List[str]:
        return ['psar']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'psar_acceleration_factor': 0.02,
         'stop_atr_mult': 1.5,
         'stoploss_multiplier': 1.5,
         'takeprofit_multiplier': 3.0,
         'tp_atr_mult': 3.0,
         'volatility_buffer': 0.2,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'psar_acceleration_factor': ParameterSpec(
                name='psar_acceleration_factor',
                min_val=0.01,
                max_val=0.2,
                default=0.02,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=5,
                max_val=100,
                default=50,
                param_type='int',
                step=1,
            ),
            'volatility_buffer': ParameterSpec(
                name='volatility_buffer',
                min_val=0,
                max_val=0.5,
                default=0.2,
                param_type='float',
                step=0.1,
            ),
            'stoploss_multiplier': ParameterSpec(
                name='stoploss_multiplier',
                min_val=1.0,
                max_val=4.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'takeprofit_multiplier': ParameterSpec(
                name='takeprofit_multiplier',
                min_val=1.0,
                max_val=4.0,
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
            signals = pd.Series(0.0, index=df.index)

            # Check for some conditions here and update 'signals' accordingly using the placeholders below

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
