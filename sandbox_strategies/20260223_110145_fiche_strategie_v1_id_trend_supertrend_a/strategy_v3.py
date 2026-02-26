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
            'leverage': 1,
            'rsi_period': 14,
            'stop_atr_mult': 2.5,
            'tp_atr_mult': 5.0,
            'warmup': 50
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
                default=2.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=5.0,
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
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)

        warmup = int(params.get('warmup', 50))

        # Extract indicator arrays as floats to avoid NaN→int conversion
        st_dir = np.asarray(indicators['supertrend']["direction"], dtype=float)
        adx_val = np.asarray(indicators['adx']["adx"], dtype=float)
        rsi_val = np.asarray(indicators['rsi'], dtype=float)
        atr_val = np.asarray(indicators['atr'], dtype=float)
        close = df["close"].values

        # Long and short entry conditions
        long_mask = (st_dir == 1) & (adx_val > 30) & (rsi_val > 60)
        short_mask = (st_dir == -1) & (adx_val > 30) & (rsi_val < 40)

        # Exit conditions
        prev_dir = np.roll(st_dir, 1)
        prev_dir[0] = np.nan
        dir_change = (prev_dir != st_dir) & ~np.isnan(prev_dir)

        prev_rsi = np.roll(rsi_val, 1)
        prev_rsi[0] = np.nan
        cross_up = (rsi_val > 50) & (prev_rsi <= 50)
        cross_down = (rsi_val < 50) & (prev_rsi >= 50)
        rsi_cross = cross_up | cross_down

        exit_mask = dir_change | (adx_val < 20) | rsi_cross

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # ATR-based SL/TP for entries
        stop_mult = params.get("stop_atr_mult", 2.5)
        tp_mult = params.get("tp_atr_mult", 5.0)

        long_entry = signals == 1.0
        short_entry = signals == -1.0

        df.loc[long_entry, "bb_stop_long"] = close[long_entry] - stop_mult * atr_val[long_entry]
        df.loc[long_entry, "bb_tp_long"] = close[long_entry] + tp_mult * atr_val[long_entry]

        df.loc[short_entry, "bb_stop_short"] = close[short_entry] + stop_mult * atr_val[short_entry]
        df.loc[short_entry, "bb_tp_short"] = close[short_entry] - tp_mult * atr_val[short_entry]

        return signals