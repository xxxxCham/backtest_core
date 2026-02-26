from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_macd')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'leverage': 1,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'rsi_period': 14,
            'stop_atr_mult': 1.5,
            'tp_atr_mult': 3.0,
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
                min_val=3,
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
                max_val=6.0,
                default=3.0,
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

        # Warm‑up period
        warmup = int(params.get('warmup', 50))

        # Indicator arrays
        macd_dict = indicators['macd']
        macd = np.nan_to_num(macd_dict["macd"])
        signal = np.nan_to_num(macd_dict["signal"])
        macd_hist = np.nan_to_num(macd_dict["histogram"])
        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])

        # Cross‑over detection
        prev_macd = np.roll(macd, 1)
        prev_signal = np.roll(signal, 1)
        prev_macd[0] = np.nan
        prev_signal[0] = np.nan
        cross_up = (macd > signal) & (prev_macd <= prev_signal)
        cross_down = (macd < signal) & (prev_macd >= prev_signal)

        # ATR increase check
        prev_atr = np.roll(atr, 1)
        prev_atr[0] = np.nan
        atr_increase = atr > prev_atr

        # RSI thresholds
        rsi_long_low = 45.0
        rsi_long_high = params.get('rsi_overbought', 70.0)
        rsi_short_low = params.get('rsi_oversold', 30.0)
        rsi_short_high = 60.0
        rsi_exit_overbought = 80.0
        rsi_exit_oversold = 20.0

        # Long and short entry masks
        long_mask = (
            cross_up
            & atr_increase
            & (rsi > rsi_long_low)
            & (rsi < rsi_long_high)
        )
        short_mask = (
            cross_down
            & atr_increase
            & (rsi > rsi_short_low)
            & (rsi < rsi_short_high)
        )

        # Exit conditions
        prev_hist = np.roll(macd_hist, 1)
        prev_hist[0] = np.nan
        hist_sign_change = (macd_hist * prev_hist) < 0
        rsi_overbought = rsi > rsi_exit_overbought
        rsi_oversold = rsi < rsi_exit_oversold
        exit_mask = hist_sign_change | rsi_overbought | rsi_oversold

        # Apply signals
        signals[exit_mask] = 0.0
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Enforce warm‑up
        signals.iloc[:warmup] = 0.0

        # ATR‑based stop‑loss / take‑profit columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        close = df["close"].values

        long_entries = signals == 1.0
        df.loc[long_entries, "bb_stop_long"] = close[long_entries] - stop_atr_mult * atr[long_entries]
        df.loc[long_entries, "bb_tp_long"] = close[long_entries] + tp_atr_mult * atr[long_entries]

        short_entries = signals == -1.0
        df.loc[short_entries, "bb_stop_short"] = close[short_entries] + stop_atr_mult * atr[short_entries]
        df.loc[short_entries, "bb_tp_short"] = close[short_entries] - tp_atr_mult * atr[short_entries]

        # Final warm‑up reset
        signals.iloc[:warmup] = 0.0
        return signals