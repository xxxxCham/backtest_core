from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='lbtc_sma_aroon_atr_trend')

    @property
    def required_indicators(self) -> List[str]:
        return ['sma', 'aroon', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'aroon_period': 14,
         'atr_period': 14,
         'leverage': 1,
         'sma_long_period': 50,
         'sma_short_period': 20,
         'stop_atr_mult': 2.2,
         'tp_atr_mult': 6.6,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'sma_short_period': ParameterSpec(
                name='sma_short_period',
                min_val=5,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'sma_long_period': ParameterSpec(
                name='sma_long_period',
                min_val=10,
                max_val=200,
                default=50,
                param_type='int',
                step=1,
            ),
            'aroon_period': ParameterSpec(
                name='aroon_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.2,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=10.0,
                default=6.6,
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
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        sma_20 = indicators['sma']
        sma_50 = indicators['sma']
        indicators['aroon']['aroon_up'] = indicators['aroon']['aroon_up']
        indicators['aroon']['aroon_down'] = indicators['aroon']['aroon_down']

        prev_sma_20 = np.roll(sma_20, 1)
        prev_sma_50 = np.roll(sma_50, 1)

        cross_above = (sma_20 > sma_50) & (prev_sma_20 <= prev_sma_50)
        cross_below = (sma_20 < sma_50) & (prev_sma_20 >= prev_sma_50)

        long_cond = cross_above & (indicators['aroon']['aroon_up'] > indicators['aroon']['aroon_down'])
        short_cond = cross_below & (indicators['aroon']['aroon_down'] > indicators['aroon']['aroon_up'])

        signals[long_cond] = 1.0
        signals[short_cond] = -1.0
        signals.iloc[:warmup] = 0.0
        return signals
