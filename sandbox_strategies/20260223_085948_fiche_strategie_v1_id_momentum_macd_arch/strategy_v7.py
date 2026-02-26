from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='macd_rsi_adx_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'rsi', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'adx_period': 14,
            'leverage': 1,
            'macd_fast_period': 12,
            'macd_signal_period': 9,
            'macd_slow_period': 26,
            'rsi_period': 14,
            'stop_atr_mult': 1.75,
            'tp_atr_mult': 2.5,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'macd_fast_period': ParameterSpec(
                name='macd_fast_period',
                min_val=5,
                max_val=30,
                default=12,
                param_type='int',
                step=1,
            ),
            'macd_slow_period': ParameterSpec(
                name='macd_slow_period',
                min_val=10,
                max_val=50,
                default=26,
                param_type='int',
                step=1,
            ),
            'macd_signal_period': ParameterSpec(
                name='macd_signal_period',
                min_val=5,
                max_val=20,
                default=9,
                param_type='int',
                step=1,
            ),
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
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.75,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.5,
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
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get("warmup", 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Ensure warmup period is zeroed
        signals.iloc[:warmup] = 0.0

        # Extract indicator arrays
        macd_vals = indicators['macd']
        macd_vals["macd"] = np.nan_to_num(macd_vals["macd"])
        macd_vals["signal"] = np.nan_to_num(macd_vals["signal"])
        histogram = np.nan_to_num(macd_vals["histogram"])
        rsi = np.nan_to_num(indicators['rsi'])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Helper cross functions
        prev_macd_line = np.roll(macd_vals["macd"], 1)
        prev_macd_signal = np.roll(macd_vals["signal"], 1)
        prev_macd_line[0] = np.nan
        prev_macd_signal[0] = np.nan
        cross_up_macd = (
            (macd_vals["macd"] > macd_vals["signal"])
            & (prev_macd_line <= prev_macd_signal)
        )
        cross_down_macd = (
            (macd_vals["macd"] < macd_vals["signal"])
            & (prev_macd_line >= prev_macd_signal)
        )

        # Entry conditions
        long_mask = (
            cross_up_macd
            & (rsi > 35)
            & (rsi < 70)
            & (adx_val > 25)
        )
        short_mask = (
            cross_down_macd
            & (rsi > 30)
            & (rsi < 60)
            & (adx_val > 25)
        )
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        prev_hist = np.roll(histogram, 1)
        prev_hist[0] = np.nan
        sign_change = (
            (histogram > 0) & (prev_hist <= 0)
            | (histogram < 0) & (prev_hist >= 0)
        )
        exit_mask = sign_change | (rsi > 80) | (rsi < 20)
        signals[exit_mask] = 0.0

        # ATR-based SL/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = float(params.get("stop_atr_mult", 1.75))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.5))

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        signals.iloc[:warmup] = 0.0
        return signals