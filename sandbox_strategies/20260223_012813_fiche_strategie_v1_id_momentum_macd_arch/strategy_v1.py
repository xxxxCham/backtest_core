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
            'macd_fast': 8,
            'macd_signal': 6,
            'macd_slow': 21,
            'rsi_period': 21,
            'stop_atr_mult': 1.75,
            'tp_atr_mult': 4.5,
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
                default=8,
                param_type='int',
                step=1,
            ),
            'macd_slow': ParameterSpec(
                name='macd_slow',
                min_val=10,
                max_val=50,
                default=21,
                param_type='int',
                step=1,
            ),
            'macd_signal': ParameterSpec(
                name='macd_signal',
                min_val=3,
                max_val=20,
                default=6,
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
                default=1.75,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
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

    def _cross_up(self, series: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """Return True where series crosses above signal."""
        cross = np.logical_and(series > signal, np.roll(series, 1) <= np.roll(signal, 1))
        cross[0] = False  # first element cannot be a cross
        return cross

    def _cross_down(self, series: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """Return True where series crosses below signal."""
        cross = np.logical_and(series < signal, np.roll(series, 1) >= np.roll(signal, 1))
        cross[0] = False
        return cross

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        warmup = int(params.get('warmup', 50))

        # Retrieve indicator arrays
        macd_vals = indicators['macd']['macd']
        macd_sig = indicators['macd']['signal']
        rsi_vals = indicators['rsi']

        # Long / short entry conditions
        long_cond = (
            self._cross_up(macd_vals, macd_sig)
            & (rsi_vals > 35)
            & (rsi_vals < 75)
        )
        short_cond = (
            self._cross_down(macd_vals, macd_sig)
            & (rsi_vals > 30)
            & (rsi_vals < 60)
        )

        signals[long_cond] = 1.0
        signals[short_cond] = -1.0

        # Apply warm‑up period
        signals.iloc[:warmup] = 0.0

        return signals