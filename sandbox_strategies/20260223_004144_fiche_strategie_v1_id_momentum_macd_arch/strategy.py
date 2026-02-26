from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_macd_optimized')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'rsi_period': 7, 'stop_atr_mult': 1.0, 'tp_atr_mult': 3.0, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=20,
                default=7,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.0,
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
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # wrap indicator arrays
        macd_arr = np.nan_to_num(indicators['macd']["macd"])
        signal_arr = np.nan_to_num(indicators['macd']["signal"])
        hist_arr = np.nan_to_num(indicators['macd']["histogram"])
        rsi_arr = np.nan_to_num(indicators['rsi'])
        atr_arr = np.nan_to_num(indicators['atr'])
        close_arr = df["close"].values

        # helper cross functions
        prev_macd = np.roll(macd_arr, 1)
        prev_signal = np.roll(signal_arr, 1)
        prev_macd[0] = np.nan
        prev_signal[0] = np.nan
        cross_up = (macd_arr > signal_arr) & (prev_macd <= prev_signal)
        cross_down = (macd_arr < signal_arr) & (prev_macd >= prev_signal)

        # entry conditions
        long_mask = cross_up & (rsi_arr > 30.0) & (rsi_arr < 60.0)
        short_mask = cross_down & (rsi_arr > 30.0) & (rsi_arr < 60.0)

        # apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # warmup protection
        signals.iloc[:warmup] = 0.0

        # prepare SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # ATR-based risk management on entry bars
        stop_mult = float(params.get("stop_atr_mult", 1.0))
        tp_mult = float(params.get("tp_atr_mult", 3.0))

        df.loc[long_mask, "bb_stop_long"] = close_arr[long_mask] - stop_mult * atr_arr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close_arr[long_mask] + tp_mult * atr_arr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close_arr[short_mask] + stop_mult * atr_arr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close_arr[short_mask] - tp_mult * atr_arr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
