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
            'macd_fast': 10,
            'macd_signal': 10,
            'macd_slow': 24,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'rsi_period': 16,
            'stop_atr_mult': 2.5,
            'tp_atr_mult': 2.5,
            'warmup': 30
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=16,
                param_type='int',
                step=1,
            ),
            'macd_fast': ParameterSpec(
                name='macd_fast',
                min_val=5,
                max_val=30,
                default=10,
                param_type='int',
                step=1,
            ),
            'macd_slow': ParameterSpec(
                name='macd_slow',
                min_val=15,
                max_val=60,
                default=24,
                param_type='int',
                step=1,
            ),
            'macd_signal': ParameterSpec(
                name='macd_signal',
                min_val=5,
                max_val=30,
                default=10,
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
                default=2.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=0.5,
                max_val=4.0,
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
    def _cross_up(series_a: np.ndarray, series_b: np.ndarray) -> np.ndarray:
        """Return mask where series_a crosses above series_b."""
        return (series_a > series_b) & (np.roll(series_a, 1) <= np.roll(series_b, 1))

    @staticmethod
    def _cross_down(series_a: np.ndarray, series_b: np.ndarray) -> np.ndarray:
        """Return mask where series_a crosses below series_b."""
        return (series_a < series_b) & (np.roll(series_a, 1) >= np.roll(series_b, 1))

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 30))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        macd_values = indicators['macd']['macd']
        signal_values = indicators['macd']['signal']
        rsi_values = indicators['rsi']

        long_mask = self._cross_up(macd_values, signal_values) & (rsi_values > 45) & (rsi_values < 65)
        short_mask = self._cross_down(macd_values, signal_values) & (rsi_values > 30) & (rsi_values < 60)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals.iloc[:warmup] = 0.0
        return signals