from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='macd_ichimoku_atr_filter_trend')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'ichimoku', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'leverage': 1,
         'stop_atr_mult': 1.0,
         'tp_atr_mult': 2.9,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=30,
                default=14,
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
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.5,
                max_val=5.0,
                default=2.9,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=20,
                max_val=100,
                default=50,
                param_type='int',
                step=1,
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

        # Extract indicator arrays
        macd_hist = np.nan_to_num(indicators['macd']["histogram"])
        senkou_a = np.nan_to_num(indicators['ichimoku']["senkou_a"])
        senkou_b = np.nan_to_num(indicators['ichimoku']["senkou_b"])
        close_arr = df["close"].values
        atr_arr = np.nan_to_num(indicators['atr'])

        # Entry conditions
        long_cond = (
            (macd_hist > 0)
            & (close_arr > senkou_a)
            & (close_arr > senkou_b)
            & ((close_arr - senkou_a) > atr_arr * 0.5)
        )
        short_cond = (
            (macd_hist < 0)
            & (close_arr < senkou_a)
            & (close_arr < senkou_b)
            & ((senkou_a - close_arr) > atr_arr * 0.5)
        )
        long_mask = long_cond
        short_mask = short_cond

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        exit_long_cond = (macd_hist < 0) | (close_arr < senkou_a)
        exit_short_cond = (macd_hist > 0) | (close_arr > senkou_a)

        # Apply exits by setting signals to 0 on bars where exit conditions are met
        signals[exit_long_cond & (signals == 1.0)] = 0.0
        signals[exit_short_cond & (signals == -1.0)] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR‑based SL/TP levels
        stop_atr_mult = float(params.get("stop_atr_mult", 1.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.9))

        # Initialize columns with NaN
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        long_entries = signals == 1.0
        short_entries = signals == -1.0

        df.loc[long_entries, "bb_stop_long"] = close_arr[long_entries] - stop_atr_mult * atr_arr[long_entries]
        df.loc[long_entries, "bb_tp_long"] = close_arr[long_entries] + tp_atr_mult * atr_arr[long_entries]

        df.loc[short_entries, "bb_stop_short"] = close_arr[short_entries] + stop_atr_mult * atr_arr[short_entries]
        df.loc[short_entries, "bb_tp_short"] = close_arr[short_entries] - tp_atr_mult * atr_arr[short_entries]
        signals.iloc[:warmup] = 0.0
        return signals
