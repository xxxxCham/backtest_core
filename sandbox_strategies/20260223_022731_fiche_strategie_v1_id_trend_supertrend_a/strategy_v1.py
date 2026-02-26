from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='trend_supertrend_v2')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'adx_period': 21,
            'leverage': 1,
            'stop_atr_mult': 1.25,
            'supertrend_atr_period': 17,
            'supertrend_multiplier': 4.0,
            'tp_atr_mult': 5.5,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'supertrend_atr_period': ParameterSpec(
                name='supertrend_atr_period',
                min_val=10,
                max_val=30,
                default=17,
                param_type='int',
                step=1,
            ),
            'supertrend_multiplier': ParameterSpec(
                name='supertrend_multiplier',
                min_val=1,
                max_val=10,
                default=4.0,
                param_type='float',
                step=0.1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=10,
                max_val=50,
                default=21,
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

        # Prepare output series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Extract indicator arrays
        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        direction = np.nan_to_num(indicators['supertrend']["direction"])
        st_val = np.nan_to_num(indicators['supertrend']["supertrend"])

        # Entry conditions
        long_mask = (direction == 1) & (close > st_val) & (adx_val > 30)
        short_mask = (direction == -1) & (close < st_val) & (adx_val > 30)

        # Exit conditions: direction change or weak ADX
        # Compute direction change without NaNs
        dir_change = np.concatenate([[False], direction[1:] != direction[:-1]])
        exit_mask = dir_change | (adx_val < 20)

        # Apply signals
        signals[exit_mask] = 0.0
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long_mask = signals == 1.0
        entry_short_mask = signals == -1.0

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

        # Final warmup enforcement
        signals.iloc[:warmup] = 0.0

        return signals