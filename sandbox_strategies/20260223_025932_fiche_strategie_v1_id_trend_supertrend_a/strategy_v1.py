from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='trend_supertrend_1h')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'adx_period': 23,
            'atr_period': 14,
            'leverage': 1,
            'stop_atr_mult': 2.75,
            'supertrend_multiplier': 2.5,
            'supertrend_period': 6,
            'tp_atr_mult': 3.5,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'supertrend_period': ParameterSpec(
                name='supertrend_period',
                min_val=5,
                max_val=20,
                default=6,
                param_type='int',
                step=1,
            ),
            'supertrend_multiplier': ParameterSpec(
                name='supertrend_multiplier',
                min_val=1.0,
                max_val=4.0,
                default=2.5,
                param_type='float',
                step=0.1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=10,
                max_val=50,
                default=23,
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
                default=2.75,
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

        # Initialise signal series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Extract indicator arrays
        direction = indicators['supertrend']['direction']
        adx_val = indicators['adx']['adx']
        atr = indicators['atr']
        close = df["close"].values

        # Validity mask to ignore NaNs in direction
        valid_dir = ~np.isnan(direction)

        # Entry logic
        long_mask = (direction == 1) & (adx_val > 25) & valid_dir
        short_mask = (direction == -1) & (adx_val > 25) & valid_dir
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit logic: direction change or weak trend
        prev_dir = np.roll(direction, 1)
        # First element has no previous value; keep mask False
        direction_change_mask = np.zeros(n, dtype=bool)
        direction_change_mask[1:] = (direction[1:] != prev_dir[1:]) & valid_dir[1:] & ~np.isnan(prev_dir[1:])
        exit_mask = direction_change_mask | (adx_val < 20)

        # Avoid exiting on the same bar as a new entry
        exit_mask &= ~((long_mask | short_mask))
        signals[exit_mask] = 0.0

        # Warmup protection
        signals[:warmup] = 0.0

        # ATR-based stop‑loss and take‑profit columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - params["stop_atr_mult"] * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + params["tp_atr_mult"] * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + params["stop_atr_mult"] * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - params["tp_atr_mult"] * atr[short_mask]

        return signals