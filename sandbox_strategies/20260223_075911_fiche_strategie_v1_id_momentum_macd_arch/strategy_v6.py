from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='macd_rsi_momentum_1h')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'leverage': 1,
            'rsi_period': 14,
            'stop_atr_mult': 1.25,
            'tp_atr_mult': 3.5,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
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
                default=1.25,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=3.5,
                param_type='float',
                step=0.1,
            ),
            'macd_fast': ParameterSpec(
                name='macd_fast',
                min_val=5,
                max_val=20,
                default=12,
                param_type='int',
                step=1,
            ),
            'macd_slow': ParameterSpec(
                name='macd_slow',
                min_val=20,
                max_val=50,
                default=26,
                param_type='int',
                step=1,
            ),
            'macd_signal': ParameterSpec(
                name='macd_signal',
                min_val=5,
                max_val=20,
                default=9,
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
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))

        # Helper functions for cross detection
        def cross_up(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return (a > b) & (np.roll(a, 1) <= np.roll(b, 1))

        def cross_down(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return (a < b) & (np.roll(a, 1) >= np.roll(b, 1))

        # Retrieve indicator arrays
        macd_vals = indicators['macd']['macd']
        signal_vals = indicators['macd']['signal']
        hist_vals = indicators['macd']['histogram']
        rsi_vals = indicators['rsi']

        # Long and short conditions
        long_cond = (
            cross_up(macd_vals, signal_vals)
            & (hist_vals > 0)
            & (rsi_vals > 55)
            & (rsi_vals < 70)
        )
        short_cond = (
            cross_down(macd_vals, signal_vals)
            & (hist_vals < 0)
            & (rsi_vals > 30)
            & (rsi_vals < 45)
        )

        signals[long_cond] = 1.0
        signals[short_cond] = -1.0
        signals.iloc[:warmup] = 0.0
        return signals