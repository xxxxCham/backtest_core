from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='3m_trend_following')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'obv']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_stop_mult': 1.5,
         'atr_tp_mult': 3.0,
         'ema_period': 20,
         'exit_threshold': 1.5,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}
    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=10,
                max_val=40,
                default=20,
                param_type='int',
                step=1,
            ),
            'atr_stop_mult': ParameterSpec(
                name='atr_stop_mult',
                min_val=1.0,
                max_val=3.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'atr_tp_mult': ParameterSpec(
                name='atr_tp_mult',
                min_val=2.5,
                max_val=4.5,
                default=3.5,
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
        obv = np.nan_to_num(indicators['obv'])
        ema = np.nan_to_num(indicators['ema'])

        long_mask = (df['close'] > ema) & (obv > 1.5*np.std(obv))
        short_mask = (df['close'] < ema) & (obv < 0.5*np.std(obv))

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        signals[:warmup] = 0.0
        signals.iloc[:warmup] = 0.0
        return signals
