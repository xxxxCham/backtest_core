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
        return ['atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'capital': 10000,
         'fees': 10,
         'leverage': 1,
         'slippage': 5,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # implement explicit LONG / SHORT / FLAT logic
        signals.iloc[:warmup] = 0.0

        if 'close' in df.columns:
            close = np.array(df['close'])
        else:
            raise ValueError('Close price data not found.')

        atr_value = indicators['atr']
        bb_stop_long_value = params.get("bb_stop_long", 1.0) * atr_value
        bb_tp_long_value = params.get("bb_tp_long", 1.5) * atr_value

        long_mask[close > close[-1]] = True
        short_mask[close < close[-1] - bb_stop_long_value] = True

        signals[(~short_mask) & (~long_mask)] = 0.0
        signals[(long_mask) | (short_mask)] = 1.0

        return signals[signals != 0].astype(int) # To get boolean output
        signals.iloc[:warmup] = 0.0
        return signals
