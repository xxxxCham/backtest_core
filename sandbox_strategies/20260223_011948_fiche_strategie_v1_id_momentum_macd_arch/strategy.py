from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_macd_with_adx_filter')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'rsi', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'leverage': 1,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'rsi_period': 14,
            'stop_atr_mult': 1.25,
            'tp_atr_mult': 4.0,
            'warmup': 50,
            # Default thresholds used in the logic
            'adx_threshold': 25,
            'adx_exit_threshold': 20,
            'rsi_short_upper': 60,
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
                default=1.25,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
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
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))

        # Boolean masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Warmup period
        signals.iloc[:warmup] = 0.0

        # Extract indicator arrays
        macd_dict = indicators['macd']
        macd_arr = np.nan_to_num(macd_dict["macd"])
        signal_line = np.nan_to_num(macd_dict["signal"])
        hist = np.nan_to_num(macd_dict["histogram"])

        rsi = np.nan_to_num(indicators['rsi'])
        adx_val = np.nan_to_num(indicators['adx']['adx'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Cross detection
        prev_macd = np.roll(macd_arr, 1)
        prev_signal = np.roll(signal_line, 1)
        prev_macd[0] = np.nan
        prev_signal[0] = np.nan
        cross_up_macd = (macd_arr > signal_line) & (prev_macd <= prev_signal)
        cross_down_macd = (macd_arr < signal_line) & (prev_macd >= prev_signal)

        # Threshold parameters with defaults
        adx_threshold = params.get('adx_threshold', 25)
        adx_exit_threshold = params.get('adx_exit_threshold', 20)
        rsi_short_upper = params.get('rsi_short_upper', 60)

        # Long entry
        long_mask = (
            cross_up_macd
            & (rsi > params["rsi_oversold"])
            & (rsi < params["rsi_overbought"])
            & (adx_val > adx_threshold)
        )

        # Short entry
        short_mask = (
            cross_down_macd
            & (rsi > params["rsi_oversold"])
            & (rsi < rsi_short_upper)
            & (adx_val > adx_threshold)
        )

        # Exit conditions
        prev_hist = np.roll(hist, 1)
        prev_hist[0] = np.nan
        hist_cross = (
            (hist > 0) & (prev_hist <= 0)
            | (hist < 0) & (prev_hist >= 0)
        )
        exit_mask = (
            hist_cross
            | (rsi > params["rsi_overbought"])
            | (rsi < params["rsi_oversold"])
            | (adx_val < adx_exit_threshold)
        )

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # ATR-based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - params["stop_atr_mult"] * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + params["tp_atr_mult"] * atr[long_mask]

        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + params["stop_atr_mult"] * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - params["tp_atr_mult"] * atr[short_mask]

        signals.iloc[:warmup] = 0.0
        return signals