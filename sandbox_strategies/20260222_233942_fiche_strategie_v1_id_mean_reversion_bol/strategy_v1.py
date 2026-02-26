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
        return ['bollinger', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'atr_min': 0.5,
            'atr_period': 14,
            'bollinger_period': 20,
            'bollinger_std': 2.5,
            'leverage': 1,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'rsi_period': 14,
            'stop_atr_mult': 1.25,
            'tp_atr_mult': 5.5,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'bollinger_std': ParameterSpec(
                name='bollinger_std',
                min_val=1.5,
                max_val=3.5,
                default=2.5,
                param_type='float',
                step=0.1,
            ),
            'atr_min': ParameterSpec(
                name='atr_min',
                min_val=0.1,
                max_val=2.0,
                default=0.5,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.25,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=10.0,
                default=5.5,
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

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        n = len(df)
        warmup = int(params.get('warmup', 50))

        # Initialize signal series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Wrap indicator arrays
        bb = indicators['bollinger']
        lower = np.nan_to_num(np.array(bb["lower"], dtype=float))
        middle = np.nan_to_num(np.array(bb["middle"], dtype=float))
        upper = np.nan_to_num(np.array(bb["upper"], dtype=float))
        rsi = np.nan_to_num(np.array(indicators['rsi'], dtype=float))
        atr = np.nan_to_num(np.array(indicators['atr'], dtype=float))
        close = np.array(df["close"].values, dtype=float)

        # Helper to detect cross any
        def cross_any(x: np.ndarray, y: np.ndarray | float) -> np.ndarray:
            # Ensure y is an array of the same shape as x
            if np.isscalar(y):
                y = np.full_like(x, y, dtype=float)
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return ((x > y) & (prev_x <= prev_y)) | ((x < y) & (prev_x >= prev_y))

        # Entry conditions
        long_mask = (close < lower) & (rsi < params["rsi_oversold"]) & (atr > params["atr_min"])
        short_mask = (close > upper) & (rsi > params["rsi_overbought"]) & (atr > params["atr_min"])

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        exit_mask = cross_any(close, middle) | cross_any(rsi, 50.0)
        signals[exit_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup]