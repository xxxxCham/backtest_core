from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='nearusdc_roc_macd_atr_momentum')

    @property
    def required_indicators(self) -> List[str]:
        return ['roc', 'macd', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'macd_fast_period': 10,
         'macd_signal_period': 9,
         'macd_slow_period': 28,
         'roc_period': 14,
         'stop_atr_mult': 1.2,
         'tp_atr_mult': 3.0,
         'warmup': 60}
    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'roc_period': ParameterSpec(
                name='roc_period',
                min_val=5,
                max_val=30,
                default=12,
                param_type='int',
                step=1,
            ),
            'macd_fast_period': ParameterSpec(
                name='macd_fast_period',
                min_val=5,
                max_val=20,
                default=12,
                param_type='int',
                step=1,
            ),
            'macd_slow_period': ParameterSpec(
                name='macd_slow_period',
                min_val=20,
                max_val=40,
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
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.4,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=2.9,
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
        # initialize masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # warmup protection
        signals.iloc[:warmup] = 0.0

        # extract needed series
        close = df["close"].values
        roc = np.nan_to_num(indicators['roc'])
        macd_dict = indicators['macd']
        macd_hist = np.nan_to_num(macd_dict["histogram"])
        atr = np.nan_to_num(indicators['atr'])

        # previous ROC for acceleration detection
        prev_roc = np.roll(roc, 1)
        prev_roc[0] = np.nan

        # entry conditions
        long_entry = (roc > 0) & (roc > prev_roc) & (macd_hist > 0)
        short_entry = (roc < 0) & (roc < prev_roc) & (macd_hist < 0)

        # exit conditions: ROC crossing zero or MACD histogram sign change
        zero = np.zeros_like(roc)
        prev_zero = np.roll(zero, 1)  # always zero, kept for symmetry
        cross_roc_up = (roc > zero) & (prev_roc <= zero)
        cross_roc_down = (roc < zero) & (prev_roc >= zero)

        prev_macd_hist = np.roll(macd_hist, 1)
        prev_macd_hist[0] = np.nan
        cross_macd_up = (macd_hist > 0) & (prev_macd_hist <= 0)
        cross_macd_down = (macd_hist < 0) & (prev_macd_hist >= 0)

        exit_mask = cross_roc_up | cross_roc_down | cross_macd_up | cross_macd_down

        # apply exit first (ensures flat on exit bars)
        signals[exit_mask] = 0.0

        # apply entries
        long_mask = long_entry & ~exit_mask
        short_mask = short_entry & ~exit_mask
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # initialize ATR SL/TP columns with NaN
        df["bb_stop_long"] = np.nan
        df["bb_tp_long"] = np.nan
        df["bb_stop_short"] = np.nan
        df["bb_tp_short"] = np.nan

        # compute SL/TP levels on entry bars
        stop_mult = float(params.get("stop_atr_mult", 1.4))
        tp_mult = float(params.get("tp_atr_mult", 2.9))

        entry_long = signals == 1.0
        entry_short = signals == -1.0

        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_mult * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_mult * atr[entry_long]

        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_mult * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_mult * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
