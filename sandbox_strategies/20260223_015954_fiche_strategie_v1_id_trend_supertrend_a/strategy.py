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
            'adx_period': 15,
            'atr_period': 14,
            'leverage': 1,
            'rsi_period': 14,
            'stop_atr_mult': 2.0,
            'supertrend_atr_period': 7,
            'supertrend_multiplier': 2.5,
            'tp_atr_mult': 4.0,
            'warmup': 50,
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
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=10,
                max_val=30,
                default=15,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
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

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        warmup = int(params.get('warmup', 50))
        signals.iloc[:warmup] = 0.0

        # Wrap indicator arrays
        supertrend_dir = np.nan_to_num(indicators['supertrend']["direction"], nan=0.0)
        adx_val = np.nan_to_num(indicators['adx']["adx"], nan=0.0)
        rsi_val = np.nan_to_num(indicators['rsi'], nan=0.0)
        atr_val = np.nan_to_num(indicators['atr'], nan=0.0)
        close = df["close"].values

        # Entry conditions
        long_mask = (supertrend_dir == 1) & (adx_val > 30) & (rsi_val > 50)
        short_mask = (supertrend_dir == -1) & (adx_val > 30) & (rsi_val < 50)

        # Exit conditions: supertrend direction change or rsi crossing 50
        prev_dir = np.roll(supertrend_dir, 1).astype(float)
        prev_dir[0] = np.nan
        dir_change = (supertrend_dir != prev_dir)

        prev_rsi = np.roll(rsi_val, 1).astype(float)
        prev_rsi[0] = np.nan
        cross_up = (rsi_val > 50) & (prev_rsi <= 50)
        cross_down = (rsi_val < 50) & (prev_rsi >= 50)
        rsi_cross = cross_up | cross_down

        exit_mask = dir_change | rsi_cross

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # SL/TP columns for ATR-based risk management
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 4.0)

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr_val[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr_val[long_mask]

        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr_val[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr_val[short_mask]

        signals.iloc[:warmup] = 0.0
        return signals