from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Regime-Adaptative ACTUSDC 1M')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'obv', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'fees': 8,
         'leverage': 1,
         'lr': 1,
         'slippage': 4,
         'stop_atr_mult': 1.5,
         'stoploss_step': 0.2,
         'takeprofit_step': 0.5,
         'tp_atr_mult': 3.0,
         'warmup': 30}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'fees': ParameterSpec(
                name='fees',
                min_val=8,
                max_val=16,
                default=8,
                param_type='int',
                step=1,
            ),
            'slippage': ParameterSpec(
                name='slippage',
                min_val=4,
                max_val=5,
                default=4,
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
        # Implement explicit LONG / SHORT / FLAT logic here
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)


        signals.iloc[:warmup] = 0.0
        signals.iloc[:warmup] = 0.0
        return signals
