from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_macd_v2')

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
            'stop_atr_mult': 1.25,
            'tp_atr_mult': 3.5,
            'warmup': 50,
            # Default RSI ranges for entry filters
            'rsi_long_low': 45,
            'rsi_long_high': 70,
            'rsi_short_low': 30,
            'rsi_short_high': 60,
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
                max_val=5.0,
                default=3.5,
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
            # Entry filter ranges
            'rsi_long_low': ParameterSpec(
                name='rsi_long_low',
                min_val=0,
                max_val=100,
                default=45,
                param_type='int',
                step=1,
            ),
            'rsi_long_high': ParameterSpec(
                name='rsi_long_high',
                min_val=0,
                max_val=100,
                default=70,
                param_type='int',
                step=1,
            ),
            'rsi_short_low': ParameterSpec(
                name='rsi_short_low',
                min_val=0,
                max_val=100,
                default=30,
                param_type='int',
                step=1,
            ),
            'rsi_short_high': ParameterSpec(
                name='rsi_short_high',
                min_val=0,
                max_val=100,
                default=60,
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
        warmup = int(params.get('warmup', 50))

        # Initialise signal series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Retrieve indicator arrays
        macd_dict = indicators['macd']
        macd_line = np.nan_to_num(macd_dict["macd"])
        signal_line = np.nan_to_num(macd_dict["signal"])
        macd_hist = np.nan_to_num(macd_dict["histogram"])

        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Helper for cross detection
        prev_macd = np.roll(macd_line, 1)
        prev_signal = np.roll(signal_line, 1)
        prev_macd[0] = np.nan
        prev_signal[0] = np.nan
        cross_up = (macd_line > signal_line) & (prev_macd <= prev_signal)
        cross_down = (macd_line < signal_line) & (prev_macd >= prev_signal)

        # Entry conditions using parameterised RSI ranges
        long_low = params.get('rsi_long_low', 45)
        long_high = params.get('rsi_long_high', 70)
        short_low = params.get('rsi_short_low', 30)
        short_high = params.get('rsi_short_high', 60)

        long_mask = cross_up & (rsi > long_low) & (rsi < long_high)
        short_mask = cross_down & (rsi > short_low) & (rsi < short_high)

        # Exit conditions
        prev_hist = np.roll(macd_hist, 1)
        prev_hist[0] = np.nan
        hist_cross = (macd_hist > 0) != (prev_hist > 0)
        exit_mask = (
            hist_cross
            | (rsi > params["rsi_overbought"])
            | (rsi < params["rsi_oversold"])
        )

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # Warmup period
        signals.iloc[:warmup] = 0.0

        # ATR-based stop/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]

        long_entry = signals == 1.0
        short_entry = signals == -1.0

        df.loc[long_entry, "bb_stop_long"] = close[long_entry] - stop_atr_mult * atr[long_entry]
        df.loc[long_entry, "bb_tp_long"] = close[long_entry] + tp_atr_mult * atr[long_entry]

        df.loc[short_entry, "bb_stop_short"] = close[short_entry] + stop_atr_mult * atr[short_entry]
        df.loc[short_entry, "bb_tp_short"] = close[short_entry] - tp_atr_mult * atr[short_entry]

        return signals