from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='trend_supertrend_rsi_filter')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'adx', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'adx_threshold_long': 35,
            'adx_threshold_exit': 25,
            'rsi_oversold': 40,
            'rsi_overbought': 60,
            'stop_atr_mult': 1.5,
            'tp_atr_mult': 4.0,
            'leverage': 1,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'adx_threshold_long': ParameterSpec(
                name='adx_threshold_long',
                min_val=20,
                max_val=50,
                default=35,
                param_type='int',
                step=1,
            ),
            'adx_threshold_exit': ParameterSpec(
                name='adx_threshold_exit',
                min_val=10,
                max_val=30,
                default=25,
                param_type='int',
                step=1,
            ),
            'rsi_oversold': ParameterSpec(
                name='rsi_oversold',
                min_val=20,
                max_val=50,
                default=40,
                param_type='int',
                step=1,
            ),
            'rsi_overbought': ParameterSpec(
                name='rsi_overbought',
                min_val=50,
                max_val=80,
                default=60,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=6.0,
                default=4.0,
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

        # Wrap indicator arrays and ensure float dtype for NaN handling
        direction = np.nan_to_num(indicators['supertrend']["direction"])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])

        # Entry conditions
        long_mask = (
            (direction == 1)
            & (adx_val > params["adx_threshold_long"])
            & (rsi > params["rsi_overbought"])
        )
        short_mask = (
            (direction == -1)
            & (adx_val > params["adx_threshold_long"])
            & (rsi < params["rsi_oversold"])
        )

        # Exit conditions
        prev_dir = np.roll(direction, 1).astype(float)
        prev_dir[0] = np.nan
        dir_change = (direction != prev_dir) & (~np.isnan(prev_dir))

        adx_exit = adx_val < params["adx_threshold_exit"]

        prev_rsi = np.roll(rsi, 1).astype(float)
        prev_rsi[0] = np.nan
        cross_up = (rsi > 50.0) & (prev_rsi <= 50.0)
        cross_down = (rsi < 50.0) & (prev_rsi >= 50.0)
        rsi_cross = cross_up | cross_down

        exit_mask = dir_change | adx_exit | rsi_cross

        # Assign entry signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Stop‑loss and take‑profit columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        close = df["close"].values
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        return signals