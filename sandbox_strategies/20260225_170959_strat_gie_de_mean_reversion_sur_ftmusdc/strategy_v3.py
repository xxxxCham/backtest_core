from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_cci_mfi_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['cci', 'mfi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'atr_median_period': 20,
            'atr_period': 14,
            'cci_overbought': 100,
            'cci_oversold': -100,
            'cci_period': 20,
            'leverage': 1,
            'mfi_overbought': 80,
            'mfi_oversold': 20,
            'mfi_period': 14,
            'stop_atr_mult': 1.5,
            'tp_atr_mult': 2.8,
            'warmup': 60,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'cci_period': ParameterSpec(
                name='cci_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'mfi_period': ParameterSpec(
                name='mfi_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_median_period': ParameterSpec(
                name='atr_median_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.8,
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
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Indicator arrays
        cci = np.nan_to_num(indicators['cci'])
        mfi = np.nan_to_num(indicators['mfi'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Helper cross functions that accept scalar thresholds
        def _to_array(x: np.ndarray, y: np.ndarray | float) -> np.ndarray:
            return np.full_like(x, y) if np.isscalar(y) else y

        def cross_up(x: np.ndarray, y: np.ndarray | float) -> np.ndarray:
            y_arr = _to_array(x, y)
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y_arr, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x > y_arr) & (prev_x <= prev_y)

        def cross_down(x: np.ndarray, y: np.ndarray | float) -> np.ndarray:
            y_arr = _to_array(x, y)
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y_arr, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x < y_arr) & (prev_x >= prev_y)

        def cross_any(x: np.ndarray, y: np.ndarray | float) -> np.ndarray:
            return cross_up(x, y) | cross_down(x, y)

        # Volatility filter: ATR above its global median
        atr_median = np.nanmedian(atr)

        # Entry conditions
        long_cond = (
            (cci <= params["cci_oversold"])
            & (mfi <= params["mfi_oversold"])
            & (atr > atr_median)
        )
        short_cond = (
            (cci >= params["cci_overbought"])
            & (mfi >= params["mfi_overbought"])
            & (atr > atr_median)
        )
        signals[long_cond] = 1.0
        signals[short_cond] = -1.0

        # Exit conditions
        cci_zero_cross_up = cross_up(cci, 0.0)
        cci_zero_cross_down = cross_down(cci, 0.0)
        mfi_cross_50 = cross_any(mfi, 50.0)

        long_exit = cci_zero_cross_up | mfi_cross_50
        short_exit = cci_zero_cross_down | mfi_cross_50

        exit_mask = (long_exit | short_exit) & (signals != 0.0)
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

        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]

        return signals