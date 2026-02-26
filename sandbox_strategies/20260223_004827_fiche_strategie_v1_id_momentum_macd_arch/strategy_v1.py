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
         'macd_fast_period': 11,
         'macd_signal_period': 10,
         'macd_slow_period': 23,
         'rsi_period': 17,
         'stop_atr_mult': 1.75,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=17,
                param_type='int',
                step=1,
            ),
            'macd_fast_period': ParameterSpec(
                name='macd_fast_period',
                min_val=5,
                max_val=30,
                default=11,
                param_type='int',
                step=1,
            ),
            'macd_slow_period': ParameterSpec(
                name='macd_slow_period',
                min_val=10,
                max_val=50,
                default=23,
                param_type='int',
                step=1,
            ),
            'macd_signal_period': ParameterSpec(
                name='macd_signal_period',
                min_val=3,
                max_val=20,
                default=10,
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

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # extract indicators
        macd_d = indicators['macd']
        indicators['macd']['macd'] = np.nan_to_num(macd_d["macd"])
        signal_line = np.nan_to_num(macd_d["signal"])
        histogram = np.nan_to_num(macd_d["histogram"])

        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # helper cross functions
        prev_macd = np.roll(indicators['macd']['macd'], 1); prev_macd[0] = np.nan
        prev_signal = np.roll(signal_line, 1); prev_signal[0] = np.nan
        cross_up = (indicators['macd']['macd'] > signal_line) & (prev_macd <= prev_signal)
        cross_down = (indicators['macd']['macd'] < signal_line) & (prev_macd >= prev_signal)

        # entry conditions
        long_mask = cross_up & (rsi > 40) & (rsi < 70)
        short_mask = cross_down & (rsi > 30) & (rsi < 60)

        # exit conditions
        prev_hist = np.roll(histogram, 1); prev_hist[0] = np.nan
        sign_change_hist = (np.sign(histogram) != np.sign(prev_hist)) & ~np.isnan(prev_hist)
        exit_mask = sign_change_hist | (rsi > 80) | (rsi < 20)

        # apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # warmup protection
        signals.iloc[:warmup] = 0.0

        # write ATR-based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = float(params.get("stop_atr_mult", 1.75))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.0))

        long_entry = signals == 1.0
        short_entry = signals == -1.0

        df.loc[long_entry, "bb_stop_long"] = close[long_entry] - stop_atr_mult * atr[long_entry]
        df.loc[long_entry, "bb_tp_long"] = close[long_entry] + tp_atr_mult * atr[long_entry]

        df.loc[short_entry, "bb_stop_short"] = close[short_entry] + stop_atr_mult * atr[short_entry]
        df.loc[short_entry, "bb_tp_short"] = close[short_entry] - tp_atr_mult * atr[short_entry]
        signals.iloc[:warmup] = 0.0
        return signals
