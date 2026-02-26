from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='trend_supertrend')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'leverage': 1,
            'stop_atr_mult': 1.75,
            'tp_atr_mult': 2.5,
            'warmup': 30
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=10,
                max_val=30,
                default=23,
                param_type='int',
                step=1,
            ),
            'supertrend_multiplier': ParameterSpec(
                name='supertrend_multiplier',
                min_val=1.0,
                max_val=3.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.75,
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

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any]
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        warmup = int(params.get('warmup', 50))

        # Extract indicator arrays
        # Use float dtype to allow NaN assignments
        direction = indicators['supertrend']["direction"].astype(float)
        adx_arr = indicators['adx']["adx"].astype(float)
        atr_arr = indicators['atr'].astype(float)
        close_arr = df["close"].values

        # Detect direction change
        prev_dir = np.roll(direction, 1)
        prev_dir[0] = np.nan
        change_dir = (direction != prev_dir) & (~np.isnan(prev_dir))

        # Entry conditions
        long_mask = (direction == 1) & (adx_arr > 25)
        short_mask = (direction == -1) & (adx_arr > 25)

        # Exit conditions
        exit_mask = change_dir | (adx_arr < 20)

        # Apply exit first to avoid overriding
        signals[exit_mask] = 0.0
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based risk levels on entry bars
        stop_atr_mult = params.get("stop_atr_mult", 1.75)
        tp_atr_mult = params.get("tp_atr_mult", 2.5)

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        df.loc[long_mask, "bb_stop_long"] = close_arr[long_mask] - stop_atr_mult * atr_arr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close_arr[long_mask] + tp_atr_mult * atr_arr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close_arr[short_mask] + stop_atr_mult * atr_arr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close_arr[short_mask] - tp_atr_mult * atr_arr[short_mask]

        signals.iloc[:warmup] = 0.0
        return signals