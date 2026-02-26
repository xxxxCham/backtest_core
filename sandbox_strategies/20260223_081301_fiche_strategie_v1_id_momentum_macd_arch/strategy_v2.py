from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_macd_adx_rsi')

    @property
    def required_indicators(self) -> List[str]:
        # ATR is required for risk management
        return ['macd', 'rsi', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'leverage': 1,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'rsi_period': 14,
            'stop_atr_mult': 1.5,
            'tp_atr_mult': 3.0,
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
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        signals.iloc[:warmup] = 0.0

        # MACD components
        macd_dict = indicators['macd']
        macd = np.nan_to_num(macd_dict["macd"])
        signal_line = np.nan_to_num(macd_dict["signal"])
        hist = np.nan_to_num(macd_dict["histogram"])

        # RSI and ADX
        rsi = np.nan_to_num(indicators['rsi'])
        adx = np.nan_to_num(indicators['adx']['adx'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Cross detection
        prev_macd = np.roll(macd, 1)
        prev_macd[0] = np.nan
        prev_signal = np.roll(signal_line, 1)
        prev_signal[0] = np.nan
        cross_up_macd = (macd > signal_line) & (prev_macd <= prev_signal)
        cross_down_macd = (macd < signal_line) & (prev_macd >= prev_signal)

        prev_hist = np.roll(hist, 1)
        prev_hist[0] = np.nan
        cross_down_hist_zero = (hist < 0) & (prev_hist >= 0)

        rsi_overbought = float(params.get("rsi_overbought", 70))
        rsi_oversold = float(params.get("rsi_oversold", 30))

        # Entry logic
        long_mask = (
            cross_up_macd
            & (rsi > rsi_oversold)
            & (rsi < rsi_overbought)
            & (adx > 25)
        )
        short_mask = (
            cross_down_macd
            & (rsi > rsi_oversold)
            & (rsi < rsi_overbought)
            & (adx > 25)
        )
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit logic
        exit_mask = (
            cross_down_hist_zero
            | (rsi > 80)
            | (rsi < 20)
            | (adx < 20)
        )
        signals[exit_mask] = 0.0

        # ATR based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = float(params.get("stop_atr_mult", 1.5))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.0))

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]

        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        signals.iloc[:warmup] = 0.0
        return signals