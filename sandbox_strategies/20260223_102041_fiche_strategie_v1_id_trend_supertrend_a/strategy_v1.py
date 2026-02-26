from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='trend_supertrend_adx')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'adx_threshold': 40,
            'leverage': 1,
            'stop_atr_mult': 1.25,
            'tp_atr_mult': 4.5,
            'warmup': 20
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'adx_threshold': ParameterSpec(
                name='adx_threshold',
                min_val=20,
                max_val=60,
                default=40,
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
                min_val=2.0,
                max_val=10.0,
                default=4.5,
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

        warmup = int(params.get('warmup', 20))

        # Extract indicator arrays as float to preserve NaNs
        st_dir = np.asarray(indicators['supertrend']["direction"], dtype=float)
        adx_val = np.asarray(indicators['adx']["adx"], dtype=float)
        atr_arr = np.asarray(indicators['atr'], dtype=float)
        close = df["close"].values

        adx_thresh = float(params.get("adx_threshold", 40.0))
        stop_mult = float(params.get("stop_atr_mult", 1.25))
        tp_mult = float(params.get("tp_atr_mult", 4.5))

        # Entry conditions
        long_mask = (st_dir == 1) & (adx_val > adx_thresh)
        short_mask = (st_dir == -1) & (adx_val > adx_thresh)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        prev_dir = np.roll(st_dir, 1)
        prev_dir[0] = np.nan
        dir_change = (prev_dir != st_dir) & (~np.isnan(prev_dir))
        adx_low = adx_val < 25.0
        exit_mask = dir_change | adx_low
        signals[exit_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = signals == 1.0
        entry_short = signals == -1.0

        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_mult * atr_arr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_mult * atr_arr[entry_long]
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_mult * atr_arr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_mult * atr_arr[entry_short]

        return signals