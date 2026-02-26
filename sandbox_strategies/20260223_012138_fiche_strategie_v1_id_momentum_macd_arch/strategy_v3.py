from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_macd_rsi_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'leverage': 1,
            'rsi_overbought': 70,
            'rsi_oversold': 35,
            'rsi_overbought_short': 60,
            'rsi_oversold_short': 30,
            'rsi_period': 7,
            'stop_atr_mult': 3.0,
            'tp_atr_mult': 5.5,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=7,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=5.5,
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
        warmup = int(params.get('warmup', 50))

        # unwrap indicators
        macd_values = np.nan_to_num(indicators['macd']['macd'])
        signal_line = np.nan_to_num(indicators['macd']['signal'])
        hist = np.nan_to_num(indicators['macd']['histogram'])
        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # cross helpers
        prev_macd = np.roll(macd_values, 1)
        prev_signal = np.roll(signal_line, 1)
        prev_macd[0] = np.nan
        prev_signal[0] = np.nan
        cross_up = (macd_values > signal_line) & (prev_macd <= prev_signal)
        cross_down = (macd_values < signal_line) & (prev_macd >= prev_signal)

        # entry conditions
        long_mask = (
            cross_up
            & (rsi > params["rsi_oversold"])
            & (rsi < params["rsi_overbought"])
        )
        short_mask = (
            cross_down
            & (rsi > params["rsi_oversold_short"])
            & (rsi < params["rsi_overbought_short"])
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # exit conditions
        prev_hist = np.roll(hist, 1)
        prev_hist[0] = np.nan
        hist_sign_change = (np.sign(hist) != np.sign(prev_hist)) & (~np.isnan(prev_hist))
        rsi_over = rsi > 80.0
        rsi_under = rsi < 20.0
        exit_mask = hist_sign_change | rsi_over | rsi_under
        signals[exit_mask] = 0.0

        # warmup period
        signals.iloc[:warmup] = 0.0

        # SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_mult = params["stop_atr_mult"]
        tp_mult = params["tp_atr_mult"]

        long_entry = signals == 1.0
        short_entry = signals == -1.0

        df.loc[long_entry, "bb_stop_long"] = close[long_entry] - stop_mult * atr[long_entry]
        df.loc[long_entry, "bb_tp_long"] = close[long_entry] + tp_mult * atr[long_entry]
        df.loc[short_entry, "bb_stop_short"] = close[short_entry] + stop_mult * atr[short_entry]
        df.loc[short_entry, "bb_tp_short"] = close[short_entry] - tp_mult * atr[short_entry]

        return signals