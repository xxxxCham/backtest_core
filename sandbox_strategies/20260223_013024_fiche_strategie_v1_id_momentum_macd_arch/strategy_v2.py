from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_macd_adx_filter')

    @property
    def required_indicators(self) -> List[str]:
        # ATR is needed for stop‑loss and take‑profit calculations
        return ['macd', 'rsi', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'adx_period': 14,
            'leverage': 1,
            'rsi_overbought': 65,
            'rsi_oversold': 35,
            'rsi_period': 14,
            'stop_atr_mult': 2.5,
            'tp_atr_mult': 5.0,
            'warmup': 50,
            # thresholds used in the logic
            'adx_entry_threshold': 25,
            'adx_exit_threshold': 20,
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
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # unwrap indicators
        macd_vals = np.nan_to_num(indicators['macd']["macd"])
        signal_line = np.nan_to_num(indicators['macd']["signal"])
        hist = np.nan_to_num(indicators['macd']["histogram"])
        rsi = np.nan_to_num(indicators['rsi'])
        adx = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # cross detection for MACD
        prev_macd = np.roll(macd_vals, 1)
        prev_signal = np.roll(signal_line, 1)
        prev_macd[0] = np.nan
        prev_signal[0] = np.nan
        cross_up_macd = (macd_vals > signal_line) & (prev_macd <= prev_signal)
        cross_down_macd = (macd_vals < signal_line) & (prev_macd >= prev_signal)

        # entry thresholds
        adx_entry = params.get("adx_entry_threshold", 25)
        adx_exit = params.get("adx_exit_threshold", 20)

        # entry conditions
        long_mask = (
            cross_up_macd
            & (rsi > params["rsi_oversold"])
            & (rsi < params["rsi_overbought"])
            & (adx > adx_entry)
        )
        short_mask = (
            cross_down_macd
            & (rsi > params["rsi_oversold"])
            & (rsi < params["rsi_overbought"])
            & (adx > adx_entry)
        )

        # exit conditions
        prev_hist = np.roll(hist, 1)
        prev_hist[0] = np.nan
        cross_any_hist = ((hist > 0) & (prev_hist <= 0)) | ((hist < 0) & (prev_hist >= 0))
        exit_mask = (
            cross_any_hist
            | (rsi > params["rsi_overbought"])
            | (rsi < params["rsi_oversold"])
            | (adx < adx_exit)
        )

        # apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - params["stop_atr_mult"] * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + params["tp_atr_mult"] * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + params["stop_atr_mult"] * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - params["tp_atr_mult"] * atr[short_mask]

        return signals