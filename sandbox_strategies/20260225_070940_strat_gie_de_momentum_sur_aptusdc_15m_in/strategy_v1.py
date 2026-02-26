from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='roc_macd_atr_momentum_15m')

    @property
    def required_indicators(self) -> List[str]:
        return ['roc', 'macd', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'macd_fast_period': 12,
         'macd_signal_period': 9,
         'macd_slow_period': 26,
         'roc_entry_thr': 0.5,
         'roc_exit_thr': 0.2,
         'roc_period': 9,
         'stop_atr_mult': 1.3,
         'tp_atr_mult': 2.21,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'roc_period': ParameterSpec(
                name='roc_period',
                min_val=5,
                max_val=30,
                default=9,
                param_type='int',
                step=1,
            ),
            'roc_entry_thr': ParameterSpec(
                name='roc_entry_thr',
                min_val=0.1,
                max_val=2.0,
                default=0.5,
                param_type='float',
                step=0.1,
            ),
            'roc_exit_thr': ParameterSpec(
                name='roc_exit_thr',
                min_val=0.05,
                max_val=1.0,
                default=0.2,
                param_type='float',
                step=0.1,
            ),
            'macd_fast_period': ParameterSpec(
                name='macd_fast_period',
                min_val=8,
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
                max_val=15,
                default=9,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.3,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=0.8,
                max_val=5.0,
                default=2.21,
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
        # extract indicator arrays with NaN handling
        roc = np.nan_to_num(indicators['roc'])
        macd_hist = np.nan_to_num(indicators['macd']["histogram"])
        atr = np.nan_to_num(indicators['atr'])

        # extract price series
        close = df["close"].values

        # parameters with defaults
        entry_thr = float(params.get("roc_entry_thr", 0.5))
        exit_thr = float(params.get("roc_exit_thr", 0.2))
        stop_mult = float(params.get("stop_atr_mult", 1.3))
        tp_mult = float(params.get("tp_atr_mult", 2.21))

        # build entry masks
        long_mask = (roc > entry_thr) & (macd_hist > 0)
        short_mask = (roc < -entry_thr) & (macd_hist < 0)

        # assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # initialize SL/TP columns
        df["bb_stop_long"] = np.nan
        df["bb_tp_long"] = np.nan
        df["bb_stop_short"] = np.nan
        df["bb_tp_short"] = np.nan

        # compute and write SL/TP levels for long entries
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]

        # compute and write SL/TP levels for short entries
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
