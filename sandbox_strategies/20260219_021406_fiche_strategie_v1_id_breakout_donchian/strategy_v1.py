from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='FICHE_STRATEGIE')

    @property
    def required_indicators(self) -> List[str]:
        return ['donchian', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=0.5,
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
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        def generate_signals(df):
            # Indicators available in this method: ['donchian', 'adx', 'atr']
            indicators = {
                'donchian': df['close'].rolling('14').mean(),
                'adx': adxl.addexceeds30(df),  # Here's the implementation of ADI indicator
                'atr': atr.calculate_atr(df)
            }

            signals = pd.Series(0.0, index=df.index, dtype=np.float64)

            for i in df.index:
                # LONG intent: close > donchian.middle and adx.addexceeds30
                if (df['close'][i] > indicators['donchian'][i]['middle']  # Here's how to access array elements
                    and indicators['adx'][i][2] >= 30):   # Here's how to call function with dict input
                  signals[i] = 1.0
                elif (df['close'][i] < indicators['donchian'][i]['lower']  # Here's how to access array elements
                      and indicators['adx'][i][2] >= 30):   # Here's how to call function with dict input
                  signals[i] = -1.0
                else:
                  signals[i] = 0.0

            return signals
        return signals
