from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='macd_supertrend_rsi')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'rsi', 'supertrend', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'leverage': 1,
            'rsi_period': 14,
            'stop_atr_mult': 2.0,
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

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)

        # Warmup period
        warmup = int(params.get('warmup', 50))
        signals.iloc[:warmup] = 0.0

        # Extract indicator arrays
        macd_dict = indicators['macd']
        macd_arr = np.nan_to_num(macd_dict["macd"])
        signal_arr = np.nan_to_num(macd_dict["signal"])
        macd_hist = np.nan_to_num(macd_dict["histogram"])
        rsi_arr = np.nan_to_num(indicators['rsi'])
        supertrend_arr = np.nan_to_num(indicators['supertrend']['supertrend'])
        atr_arr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Cross detection for MACD
        prev_macd = np.roll(macd_arr, 1)
        prev_signal = np.roll(signal_arr, 1)
        prev_macd[0] = np.nan
        prev_signal[0] = np.nan
        cross_up = (macd_arr > signal_arr) & (prev_macd <= prev_signal)
        cross_down = (macd_arr < signal_arr) & (prev_macd >= prev_signal)

        # Thresholds for RSI (hard‑coded to match strategy intent)
        rsi_long_lower = 45
        rsi_long_upper = 65
        rsi_short_lower = 35
        rsi_short_upper = 55
        rsi_exit_high = 80
        rsi_exit_low = 20

        # Entry conditions
        long_mask = (
            cross_up
            & (close > supertrend_arr)
            & (rsi_arr > rsi_long_lower)
            & (rsi_arr < rsi_long_upper)
        )
        short_mask = (
            cross_down
            & (close < supertrend_arr)
            & (rsi_arr > rsi_short_lower)
            & (rsi_arr < rsi_short_upper)
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        hist_prev = np.roll(macd_hist, 1)
        hist_prev[0] = np.nan
        hist_sign_change = (
            (macd_hist > 0) & (hist_prev <= 0)
            | (macd_hist < 0) & (hist_prev >= 0)
        )
        rsi_exit = (rsi_arr > rsi_exit_high) | (rsi_arr < rsi_exit_low)
        exit_mask = hist_sign_change | rsi_exit
        signals[exit_mask] = 0.0

        # ATR-based stop‑loss / take‑profit (for reference)
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = signals == 1.0
        entry_short = signals == -1.0

        df.loc[entry_long, "bb_stop_long"] = (
            close[entry_long] - params["stop_atr_mult"] * atr_arr[entry_long]
        )
        df.loc[entry_long, "bb_tp_long"] = (
            close[entry_long] + params["tp_atr_mult"] * atr_arr[entry_long]
        )

        df.loc[entry_short, "bb_stop_short"] = (
            close[entry_short] + params["stop_atr_mult"] * atr_arr[entry_short]
        )
        df.loc[entry_short, "bb_tp_short"] = (
            close[entry_short] - params["tp_atr_mult"] * atr_arr[entry_short]
        )

        signals.iloc[:warmup] = 0.0
        return signals