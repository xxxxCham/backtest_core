from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_macd_rsi_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'atr_period': 14,
            'leverage': 1,
            'macd_fast': 14,
            'macd_signal': 7,
            'macd_slow': 20,
            'rsi_period': 21,
            'stop_atr_mult': 2.0,
            'tp_atr_mult': 2.5,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=21,
                param_type='int',
                step=1,
            ),
            'macd_fast': ParameterSpec(
                name='macd_fast',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'macd_slow': ParameterSpec(
                name='macd_slow',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'macd_signal': ParameterSpec(
                name='macd_signal',
                min_val=3,
                max_val=15,
                default=7,
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
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.5,
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

    @staticmethod
    def _cross_up(series: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """Return True where series crosses above signal."""
        cross = (np.roll(series, 1) <= signal) & (series > signal)
        cross[0] = False  # first element has no previous value
        return cross

    @staticmethod
    def _cross_down(series: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """Return True where series crosses below signal."""
        cross = (np.roll(series, 1) >= signal) & (series < signal)
        cross[0] = False
        return cross

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))

        # Initialize masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Compute entry conditions
        long_mask = (
            self._cross_up(indicators['macd']['macd'], indicators['macd']['signal'])
            & (indicators['rsi'] > 40)
            & (indicators['rsi'] < 80)
        )
        short_mask = (
            self._cross_down(indicators['macd']['macd'], indicators['macd']['signal'])
            & (indicators['rsi'] > 30)
            & (indicators['rsi'] < 60)
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Zero out signals during warm‑up period
        signals.iloc[:warmup] = 0.0
        return signals