from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='trend_supertrend_sma_filter')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'adx', 'sma', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'adx_exit_threshold': 20,
            'adx_threshold': 25,
            'leverage': 1,
            'sma_period': 50,
            'stop_atr_mult': 2.75,
            'tp_atr_mult': 5.5,
            'warmup': 50
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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
                max_val=10.0,
                default=5.5,
                param_type='float',
                step=0.1,
            ),
            'sma_period': ParameterSpec(
                name='sma_period',
                min_val=10,
                max_val=200,
                default=50,
                param_type='int',
                step=1,
            ),
            'adx_threshold': ParameterSpec(
                name='adx_threshold',
                min_val=10,
                max_val=40,
                default=25,
                param_type='int',
                step=1,
            ),
            'adx_exit_threshold': ParameterSpec(
                name='adx_exit_threshold',
                min_val=10,
                max_val=30,
                default=20,
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

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        n = len(df)
        warmup = int(params.get('warmup', 50))
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Extract indicator arrays
        direction = np.array(indicators['supertrend']["direction"], dtype=float)
        adx_val = np.array(indicators['adx']["adx"], dtype=float)
        sma_val = np.array(indicators['sma'], dtype=float)
        close = df["close"].values
        atr = np.array(indicators['atr'], dtype=float)

        # Entry conditions
        long_mask = (
            (direction == 1)
            & (adx_val > params["adx_threshold"])
            & (close > sma_val)
        )
        short_mask = (
            (direction == -1)
            & (adx_val > params["adx_threshold"])
            & (close < sma_val)
        )

        # Exit conditions: direction change or ADX below threshold
        # Compute direction change without NaN assignment
        dir_change = np.concatenate(
            [[False], direction[1:] != direction[:-1]]
        )
        exit_mask = dir_change | (adx_val < params["adx_exit_threshold"])
        # Avoid overriding entry signals
        exit_mask &= ~(long_mask | short_mask)

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR‑based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long_mask = long_mask
        entry_short_mask = short_mask

        df.loc[entry_long_mask, "bb_stop_long"] = (
            close[entry_long_mask] - params["stop_atr_mult"] * atr[entry_long_mask]
        )
        df.loc[entry_long_mask, "bb_tp_long"] = (
            close[entry_long_mask] + params["tp_atr_mult"] * atr[entry_long_mask]
        )
        df.loc[entry_short_mask, "bb_stop_short"] = (
            close[entry_short_mask] + params["stop_atr_mult"] * atr[entry_short_mask]
        )
        df.loc[entry_short_mask, "bb_tp_short"] = (
            close[entry_short_mask] - params["tp_atr_mult"] * atr[entry_short_mask]
        )

        return signals