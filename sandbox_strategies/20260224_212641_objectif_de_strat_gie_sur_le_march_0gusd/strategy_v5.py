from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Scalp ADX Bollinger RSI MACD Vortex Aroon ICHIMOKU')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'ema', 'adx', 'macd', 'vortex', 'ichimoku', 'obv', 'williams_r']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': '',
         'leverage': 1,
         'obv_diff_days': 0,
         'obv_period': 8,
         'roc_fast_length': 12,
         'roc_signal_deviation': 3,
         'roc_signal_length': 9,
         'roc_slow_length': 24,
         'stop_atr_mult': 1.5,
         'stop_trig_mult': 1.3,
         'takeprofit_trig_mult': 2,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'obv_period': ParameterSpec(
                name='obv_period',
                min_val=4,
                max_val=50,
                default=8,
                param_type='int',
                step=1,
            ),
            'roc_fast_length': ParameterSpec(
                name='roc_fast_length',
                min_val=10,
                max_val=70,
                default=12,
                param_type='int',
                step=1,
            ),
            'roc_slow_length': ParameterSpec(
                name='roc_slow_length',
                min_val=40,
                max_val=80,
                default=24,
                param_type='int',
                step=1,
            ),
            'roc_signal_length': ParameterSpec(
                name='roc_signal_length',
                min_val=9,
                max_val=70,
                default=9,
                param_type='int',
                step=1,
            ),
            'roc_signal_deviation': ParameterSpec(
                name='roc_signal_deviation',
                min_val=-5,
                max_val=10,
                default=3,
                param_type='float',
                step=0.1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=4,
                max_val=79,
                default=8,
                param_type='int',
                step=1,
            ),
            'stop_trig_mult': ParameterSpec(
                name='stop_trig_mult',
                min_val=1.0,
                max_val=2.5,
                default=1.3,
                param_type='float',
                step=0.1,
            ),
            'takeprofit_trig_mult': ParameterSpec(
                name='takeprofit_trig_mult',
                min_val=2.0,
                max_val=4.0,
                default=2.0,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Implement explicit LONG / SHORT / FLAT logic here
        signals.iloc[:warmup] = 0.0
        signals.iloc[:warmup] = 0.0
        return signals
