from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='trend_supertrend_modified')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'adx_exit_threshold': 20,
            'adx_threshold': 25,
            'leverage': 1,
            'stop_atr_mult': 1.0,
            'tp_atr_mult': 2.5,
            'warmup': 30
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)

        # Warm‑up period
        warmup = int(params.get('warmup', 50))

        # Prepare columns for stop/TP (they may be used downstream)
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Extract indicator arrays
        st = indicators['supertrend']
        adx_d = indicators['adx']
        atr = np.nan_to_num(indicators['atr'])

        # Ensure numeric arrays
        direction = np.nan_to_num(st["direction"]).astype(float)
        adx_val = np.nan_to_num(adx_d["adx"]).astype(float)

        close = df["close"].values.astype(float)

        # Parameter values
        adx_thresh = float(params.get("adx_threshold", 25))
        adx_exit_thresh = float(params.get("adx_exit_threshold", 20))
        stop_mult = float(params.get("stop_atr_mult", 1.0))
        tp_mult = float(params.get("tp_atr_mult", 2.5))

        # Entry masks
        long_mask = (direction == 1) & (adx_val > adx_thresh)
        short_mask = (direction == -1) & (adx_val > adx_thresh)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ATR‑based stop/TP
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]

        # Exit conditions
        prev_dir = np.roll(direction, 1).astype(float)
        prev_dir[0] = np.nan
        dir_change = (direction != prev_dir) & ~np.isnan(prev_dir)

        exit_mask = dir_change | (adx_val < adx_exit_thresh)
        signals[exit_mask] = 0.0

        # Protect warm‑up
        signals.iloc[:warmup] = 0.0

        return signals