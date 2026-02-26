from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='supertrend_rsi_trend_filter')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'leverage': 1,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'rsi_period': 14,
            'stop_atr_mult': 1.0,
            'tp_atr_mult': 3.5,
            'warmup': 30
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
                default=1.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.5,
                max_val=6.0,
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
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Warm‑up period
        warmup = int(params.get('warmup', 50))

        # Extract indicator arrays as floats
        st = indicators['supertrend']
        st_dir = np.array(st["direction"], dtype=float)          # 1 / -1 / NaN
        st_line = np.array(st["supertrend"], dtype=float)       # float series
        rsi = np.array(indicators['rsi'], dtype=float)
        atr = np.array(indicators['atr'], dtype=float)

        close = df["close"].values

        # Parameters for thresholds
        rsi_long_thr = float(params.get("rsi_long_threshold", 60.0))
        rsi_short_thr = float(params.get("rsi_short_threshold", 40.0))
        rsi_cross_thr = float(params.get("rsi_cross_threshold", 50.0))
        stop_atr_mult = float(params.get("stop_atr_mult", 1.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.5))

        # Long entry conditions
        long_mask = (
            (close > st_line)
            & (st_dir == 1)
            & (rsi > rsi_long_thr)
        )

        # Short entry conditions
        short_mask = (
            (close < st_line)
            & (st_dir == -1)
            & (rsi < rsi_short_thr)
        )

        # Exit conditions
        prev_dir = np.roll(st_dir, 1).astype(float)
        prev_dir[0] = np.nan
        dir_change = (st_dir != prev_dir) & (~np.isnan(st_dir)) & (~np.isnan(prev_dir))

        prev_rsi = np.roll(rsi, 1).astype(float)
        prev_rsi[0] = np.nan
        rsi_cross_down = (rsi < rsi_cross_thr) & (prev_rsi >= rsi_cross_thr)
        rsi_cross_up = (rsi > rsi_cross_thr) & (prev_rsi <= rsi_cross_thr)

        long_exit = dir_change | rsi_cross_down
        short_exit = dir_change | rsi_cross_up

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[long_exit] = 0.0
        signals[short_exit] = 0.0

        # Enforce warm‑up
        signals.iloc[:warmup] = 0.0

        # Stop‑loss and take‑profit columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long_mask = (signals == 1.0)
        entry_short_mask = (signals == -1.0)

        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_atr_mult * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_atr_mult * atr[entry_long_mask]

        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_atr_mult * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_atr_mult * atr[entry_short_mask]

        # Final warm‑up enforcement
        signals.iloc[:warmup] = 0.0
        return signals