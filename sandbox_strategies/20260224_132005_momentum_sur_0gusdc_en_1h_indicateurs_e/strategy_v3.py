from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Snake Case Name')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'aroon', 'macd']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'aroon_length': 50,
         'aroon_middleband': 70,
         'bollinger_lowerband': None,
         'bollinger_upperband': None,
         'default_params_specs': {'aroon_length': {'max': 80, 'min': 5},
                                  'ema_length': {'max': 30, 'min': 5},
                                  'sar_length': {'max': 20, 'min': 10}},
         'ema_length': 30,
         'ema_middleband': 30,
         'leverage': 1,
         'macd_fast_length': 12,
         'macd_fast_period': 12,
         'macd_slow_length': 26,
         'macd_slow_period': 26,
         'sar_length': 20,
         'sma_length': 9,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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
        def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, ParameterSpec]) -> pd.Series:
            """Builds the trading strategy."""

            # Generate signals for long/short positions based on inputs and logic within self.generate_signals method
        signals.iloc[:warmup] = 0.0
        return signals
