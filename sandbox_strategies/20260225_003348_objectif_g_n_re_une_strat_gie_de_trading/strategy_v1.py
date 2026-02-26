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
        return ['ema', 'obv', 'stochastic']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'ema_period': 10,
         'leverage': 1,
         'obv_lookback': 40,
         'roc_period': 9,
         'roc_threshold': 0.25,
         'stochastic_d_length': 3,
         'stochastic_k_length': 3,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=3,
                max_val=60,
                default=10,
                param_type='int',
                step=1,
            ),
            'obv_lookback': ParameterSpec(
                name='obv_lookback',
                min_val=20,
                max_val=80,
                default=40,
                param_type='int',
                step=1,
            ),
            'stochastic_k_length': ParameterSpec(
                name='stochastic_k_length',
                min_val=3,
                max_val=9,
                default=3,
                param_type='int',
                step=1,
            ),
            'stochastic_d_length': ParameterSpec(
                name='stochastic_d_length',
                min_val=3,
                max_val=15,
                default=3,
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
        def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
            n = len(df)
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)
            long_mask = np.zeros(n, dtype=bool)
            short_mask = np.zeros(n, dtype=bool)
            # implement explicit LONG / SHORT / FLAT logic
            # warmup protection
            warmup = int(params.get("warmup", 50))
            signals.iloc[:warmup] = 0.0
            # Write SL/TP columns into df if using ATR-based risk management
            return signals
        signals.iloc[:warmup] = 0.0
        return signals
