from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_macd_rsi')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'leverage': 1,
            'rsi_overbought': 65,
            'rsi_oversold': 35,
            'rsi_period': 14,
            'stop_atr_mult': 1.5,
            'tp_atr_mult': 4.0,
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
                max_val=3.0,
                default=1.5,
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

        # unwrap indicator arrays
        macd_dict = indicators['macd']
        macd_line = np.nan_to_num(macd_dict["macd"])
        signal_line = np.nan_to_num(macd_dict["signal"])
        hist = np.nan_to_num(macd_dict["histogram"])
        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])

        # cross helpers
        prev_macd = np.roll(macd_line, 1)
        prev_macd[0] = np.nan
        prev_signal = np.roll(signal_line, 1)
        prev_signal[0] = np.nan
        cross_up = (macd_line > signal_line) & (prev_macd <= prev_signal)
        cross_down = (macd_line < signal_line) & (prev_macd >= prev_signal)

        # long and short entry conditions
        rsi_oversold = params.get('rsi_oversold', 35)
        rsi_overbought = params.get('rsi_overbought', 65)
        long_mask = cross_up & (rsi > rsi_oversold) & (rsi < rsi_overbought)
        short_mask = cross_down & (rsi > rsi_oversold) & (rsi < rsi_overbought)

        # exit conditions
        prev_hist = np.roll(hist, 1)
        prev_hist[0] = np.nan
        hist_sign_change = (hist * prev_hist < 0)
        rsi_exit_overbought = 80
        rsi_exit_oversold = 20
        rsi_exit = (rsi > rsi_exit_overbought) | (rsi < rsi_exit_oversold)
        exit_mask = hist_sign_change | rsi_exit

        # assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # warmup period
        signals.iloc[:warmup] = 0.0

        # ATR based stop‑loss and take‑profit
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        close = df["close"].values
        long_entry = signals == 1.0
        short_entry = signals == -1.0

        df.loc[long_entry, "bb_stop_long"] = close[long_entry] - params["stop_atr_mult"] * atr[long_entry]
        df.loc[long_entry, "bb_tp_long"] = close[long_entry] + params["tp_atr_mult"] * atr[long_entry]

        df.loc[short_entry, "bb_stop_short"] = close[short_entry] + params["stop_atr_mult"] * atr[short_entry]
        df.loc[short_entry, "bb_tp_short"] = close[short_entry] - params["tp_atr_mult"] * atr[short_entry]

        return signals