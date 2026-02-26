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
        return ['rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'Leverage': '1',
         'RSILimit': 14,
         'StopAtrMultiplier': '1.5',
         'TakeProfitAtRsiLimitAtm': '3',
         'WarmupPeriodInTrades': '50',
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'RSILimit': ParameterSpec(
                name='RSILimit',
                min_val=5,
                max_val=50,
                default='14',
                param_type='int',
                step=1,
            ),
            'StopAtrMultiplier': ParameterSpec(
                name='StopAtrMultiplier',
                min_val=0.5,
                max_val=4,
                default='1.5',
                param_type='float',
                step=0.1,
            ),
            'TakeProfitAtRsiLimitAtm': ParameterSpec(
                name='TakeProfitAtRsiLimitAtm',
                min_val=0,
                max_val=30,
                default='2',
                param_type='float',
                step=0.1,
            ),
            'WarmupPeriodInTrades': ParameterSpec(
                name='WarmupPeriodInTrades',
                min_val=0,
                max_val=100,
                default='50',
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
        # Add your code here to implement the logic for generating trading signals based on rsi and atr
        pass  # This line is a placeholder and should be removed when you add your actual logic
        signals.iloc[:warmup] = 0.0
        return signals
