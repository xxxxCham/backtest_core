from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='trend_supertrend_ema_adx')

    @property
    def required_indicators(self) -> List[str]:
        # Include ATR because it is used for risk management
        return ['supertrend', 'adx', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'ema_period': 50,
            'leverage': 1,
            'stop_atr_mult': 1.5,
            'tp_atr_mult': 2.5,
            'warmup': 50
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=10,
                max_val=200,
                default=50,
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

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))

        # Helper cross functions
        def cross_up(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x > y) & (prev_x <= prev_y)

        def cross_down(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x < y) & (prev_x >= prev_y)

        # Extract indicator arrays
        close = df["close"].values
        ema = np.nan_to_num(indicators['ema'])
        supertrend_dir = np.nan_to_num(indicators['supertrend']["direction"])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])

        # Entry conditions
        long_mask = (
            (supertrend_dir == 1)
            & (adx_val > 25)
            & (close > ema)
        )
        short_mask = (
            (supertrend_dir == -1)
            & (adx_val > 25)
            & (close < ema)
        )

        # Exit conditions
        cross_up_close_ema = cross_up(close, ema)
        cross_down_close_ema = cross_down(close, ema)

        exit_long_mask = (
            (supertrend_dir == -1)
            | (adx_val < 20)
            | cross_down_close_ema
        )
        exit_short_mask = (
            (supertrend_dir == 1)
            | (adx_val < 20)
            | cross_up_close_ema
        )

        # Apply entry signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Apply exit signals (flatten)
        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = float(params.get("stop_atr_mult", 1.5))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.5))

        entry_long_mask = signals == 1.0
        entry_short_mask = signals == -1.0

        df.loc[entry_long_mask, "bb_stop_long"] = (
            close[entry_long_mask] - stop_atr_mult * atr[entry_long_mask]
        )
        df.loc[entry_long_mask, "bb_tp_long"] = (
            close[entry_long_mask] + tp_atr_mult * atr[entry_long_mask]
        )
        df.loc[entry_short_mask, "bb_stop_short"] = (
            close[entry_short_mask] + stop_atr_mult * atr[entry_short_mask]
        )
        df.loc[entry_short_mask, "bb_tp_short"] = (
            close[entry_short_mask] - tp_atr_mult * atr[entry_short_mask]
        )

        signals.iloc[:warmup] = 0.0
        return signals