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
        return {'leverage': 1,
         'macd_fast_period': 14,
         'macd_signal_period': 6,
         'macd_slow_period': 32,
         'rsi_period': 21,
         'stop_atr_mult': 2.0,
         'tp_atr_mult': 6.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'macd_fast_period': ParameterSpec(
                name='macd_fast_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'macd_slow_period': ParameterSpec(
                name='macd_slow_period',
                min_val=15,
                max_val=50,
                default=32,
                param_type='int',
                step=1,
            ),
            'macd_signal_period': ParameterSpec(
                name='macd_signal_period',
                min_val=3,
                max_val=10,
                default=6,
                param_type='int',
                step=1,
            ),
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=21,
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
                default=6.0,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        macd_vals = np.nan_to_num(indicators['macd']["macd"])
        signal_vals = np.nan_to_num(indicators['macd']["signal"])
        hist_vals = np.nan_to_num(indicators['macd']["histogram"])
        rsi_vals = np.nan_to_num(indicators['rsi'])
        atr_vals = np.nan_to_num(indicators['atr'])
        close_vals = df["close"].values

        prev_macd = np.roll(macd_vals, 1)
        prev_signal = np.roll(signal_vals, 1)
        prev_macd[0] = np.nan
        prev_signal[0] = np.nan

        cross_up = (macd_vals > signal_vals) & (prev_macd <= prev_signal)
        cross_down = (macd_vals < signal_vals) & (prev_macd >= prev_signal)

        long_mask = cross_up & (rsi_vals > 35) & (rsi_vals < 65)
        short_mask = cross_down & (rsi_vals > 30) & (rsi_vals < 60)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        prev_hist = np.roll(hist_vals, 1)
        prev_hist[0] = np.nan
        hist_cross = (np.sign(hist_vals) != np.sign(prev_hist))
        exit_mask = hist_cross | (rsi_vals > 80) | (rsi_vals < 20)
        signals[exit_mask] = 0.0

        signals.iloc[:warmup] = 0.0

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_mult = float(params.get("stop_atr_mult", 2.0))
        tp_mult = float(params.get("tp_atr_mult", 6.0))

        df.loc[long_mask, "bb_stop_long"] = close_vals[long_mask] - stop_mult * atr_vals[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close_vals[long_mask] + tp_mult * atr_vals[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close_vals[short_mask] + stop_mult * atr_vals[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close_vals[short_mask] - tp_mult * atr_vals[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
