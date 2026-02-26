from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_bollinger_rsi_v3')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'leverage': 1,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'rsi_period': 14,
            'stop_atr_mult': 2.0,
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
                default=2.0,
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
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any]
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        warmup = int(params.get('warmup', 50))

        # --- Indicator extraction -------------------------------------------------
        close = df["close"].values
        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])

        bb = indicators['bollinger']
        lower = np.nan_to_num(bb["lower"])
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])

        # rolling mean of ATR with window 20
        window = 20
        atr_sma = np.convolve(atr, np.ones(window) / window, mode="same")

        # --- Helper ---------------------------------------------------------------
        def cross_any(x: np.ndarray, y: np.ndarray | float) -> np.ndarray:
            """
            Detect crossing between two series or a series and a constant.
            Handles scalar `y` by broadcasting to the shape of `x`.
            """
            if np.isscalar(y):
                y_arr = np.full_like(x, y)
            else:
                y_arr = y

            prev_x = np.roll(x, 1)
            prev_y = np.roll(y_arr, 1)

            # first element cannot be compared
            prev_x[0] = np.nan
            prev_y[0] = np.nan

            return ((x > y_arr) & (prev_x <= prev_y)) | ((x < y_arr) & (prev_x >= prev_y))

        # --- Entry conditions -----------------------------------------------------
        long_mask = (
            (close < lower)
            & (rsi < params["rsi_oversold"])
            & (atr < atr_sma)
        )
        short_mask = (
            (close > upper)
            & (rsi > params["rsi_overbought"])
            & (atr < atr_sma)
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # --- Exit conditions -------------------------------------------------------
        exit_mask = cross_any(close, middle) | cross_any(rsi, 50.0)
        signals[exit_mask] = 0.0

        # Ensure no signals during warmup
        signals.iloc[:warmup] = 0.0

        # --- ATR-based SL/TP (stored for reference) -------------------------------
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - params["stop_atr_mult"] * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + params["tp_atr_mult"] * atr[long_mask]

        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + params["stop_atr_mult"] * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - params["tp_atr_mult"] * atr[short_mask]

        return signals