from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='supertrend_rsi_atr_30m')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'atr_period': 14,
            'leverage': 1,
            'rsi_period': 14,
            'stop_atr_mult': 2.4,
            'supertrend_multiplier': 3.0,
            'supertrend_period': 10,
            'tp_atr_mult': 5.8,
            'warmup': 30,
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
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.4,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=10.0,
                default=5.8,
                param_type='float',
                step=0.1,
            ),
            'supertrend_period': ParameterSpec(
                name='supertrend_period',
                min_val=5,
                max_val=20,
                default=10,
                param_type='int',
                step=1,
            ),
            'supertrend_multiplier': ParameterSpec(
                name='supertrend_multiplier',
                min_val=1.0,
                max_val=4.0,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=50,
                default=14,
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

        # unwrap indicators (use float arrays to allow NaN handling)
        rsi = np.nan_to_num(indicators['rsi'], nan=0.0)
        atr = np.nan_to_num(indicators['atr'], nan=0.0)
        st = np.nan_to_num(indicators['supertrend']["direction"], nan=0.0).astype(float)
        close = df["close"].values

        # compute 20‑period ATR average via convolution
        atr_20_avg = np.convolve(atr, np.ones(20) / 20, mode="same")

        # entry conditions
        long_mask = (st == 1) & (rsi > 60) & (atr > 1.5 * atr_20_avg)
        short_mask = (st == -1) & (rsi < 40) & (atr > 1.5 * atr_20_avg)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # exit conditions
        prev_st = np.roll(st, 1).astype(float)
        prev_st[0] = np.nan
        direction_change = (st != prev_st)

        prev_rsi = np.roll(rsi, 1).astype(float)
        prev_rsi[0] = np.nan
        cross_up = (rsi > 50) & (prev_rsi <= 50)
        cross_down = (rsi < 50) & (prev_rsi >= 50)
        rsi_cross = cross_up | cross_down

        exit_mask = (direction_change | rsi_cross) & (~(long_mask | short_mask))
        signals[exit_mask] = 0.0

        # warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR‑based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params.get("stop_atr_mult", 2.4)
        tp_atr_mult = params.get("tp_atr_mult", 5.8)

        if long_mask.any():
            df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
            df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        if short_mask.any():
            df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
            df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        signals.iloc[:warmup] = 0.0
        return signals