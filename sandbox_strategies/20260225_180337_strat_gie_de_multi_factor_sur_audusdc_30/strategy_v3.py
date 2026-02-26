from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='macd_adx_atr_30m_audusdc')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'atr_period': 14,
         'leverage': 1,
         'macd_fast_period': 12,
         'macd_signal_period': 9,
         'macd_slow_period': 26,
         'stop_atr_mult': 1.3,
         'tp_atr_mult': 3.6,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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
                max_val=50,
                default=26,
                param_type='int',
                step=1,
            ),
            'macd_signal_period': ParameterSpec(
                name='macd_signal_period',
                min_val=3,
                max_val=15,
                default=9,
                param_type='int',
                step=1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
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
                default=1.3,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=3.6,
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
        # boolean masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # extract indicators
        macd_vals = np.nan_to_num(indicators['macd']["macd"])
        macd_sig = np.nan_to_num(indicators['macd']["signal"])
        adx_vals = np.nan_to_num(indicators['adx']["adx"])
        atr_vals = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # helper cross functions
        prev_macd = np.roll(macd_vals, 1)
        prev_sig = np.roll(macd_sig, 1)
        prev_macd[0] = np.nan
        prev_sig[0] = np.nan
        cross_up = (macd_vals > macd_sig) & (prev_macd <= prev_sig)
        cross_down = (macd_vals < macd_sig) & (prev_macd >= prev_sig)

        # long entry: macd above signal and ADX > 25
        long_mask = (macd_vals > macd_sig) & (adx_vals > 25)

        # short entry: macd below signal and ADX > 25
        short_mask = (macd_vals < macd_sig) & (adx_vals > 25)

        # exit long when MACD crosses below signal
        exit_long_mask = cross_down

        # exit short when MACD crosses above signal
        exit_short_mask = cross_up

        # apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0

        # warmup protection
        signals.iloc[:warmup] = 0.0

        # write ATR-based SL/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_mult = params.get("stop_atr_mult", 1.3)
        tp_mult = params.get("tp_atr_mult", 3.6)

        df.loc[signals == 1.0, "bb_stop_long"] = close[signals == 1.0] - stop_mult * atr_vals[signals == 1.0]
        df.loc[signals == 1.0, "bb_tp_long"] = close[signals == 1.0] + tp_mult * atr_vals[signals == 1.0]

        df.loc[signals == -1.0, "bb_stop_short"] = close[signals == -1.0] + stop_mult * atr_vals[signals == -1.0]
        df.loc[signals == -1.0, "bb_tp_short"] = close[signals == -1.0] - tp_mult * atr_vals[signals == -1.0]
        signals.iloc[:warmup] = 0.0
        return signals
