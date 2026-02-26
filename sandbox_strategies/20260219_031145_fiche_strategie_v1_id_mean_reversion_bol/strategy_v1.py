from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_bollinger_rsi')

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 20,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=2.25,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=4.5,
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
        def generate_signals(indicators):
            # LONG intent conditions
            close_bollinger = indicators['bollinger']['lower']
            rsi = indicators['rsi']

            long_signal = np.zeros(len(df), dtype=np.float64)  # initialize signal vector to zeros

            for i, row in df[close_bollinger].iterrows():
                if (row > close_bollinger).all() and rsi < 50:   # check conditions for long position
                    long_signal[i] = -1.0                               # assign signal value as -1.0

            # SHORT intent conditions
            close_bollinger = indicators['bollinger']['upper']
            rsi = indicators['rsi']

            short_signal = np.zeros(len(df), dtype=np.float64)  # initialize signal vector to zeros

            for i, row in df[close_bollinger].iterrows():
                if (row < close_bollinger).all() and rsi > 70:   # check conditions for short position
                    short_signal[i] = -1.0                              # assign signal value as -1.0

            return long_signal, short_signal
        return signals
