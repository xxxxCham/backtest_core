from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='macd_rsi_atr_vol_filter')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_threshold': 0.5,
         'leverage': 1,
         'rsi_period': 14,
         'stop_atr_mult': 2.0,
         'tp_atr_mult': 6.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_threshold': ParameterSpec(
                name='atr_threshold',
                min_val=0.1,
                max_val=2.0,
                default=0.5,
                param_type='float',
                step=0.1,
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

        signals.iloc[:warmup] = 0.0

        macd_vals = np.nan_to_num(indicators['macd']["macd"])
        signal_vals = np.nan_to_num(indicators['macd']["signal"])
        hist_vals = np.nan_to_num(indicators['macd']["histogram"])
        rsi_vals = np.nan_to_num(indicators['rsi'])
        atr_vals = np.nan_to_num(indicators['atr'])
        close_vals = df["close"].values

        def cross_up(x, y):
            px = np.roll(x, 1)
            py = np.roll(y, 1)
            px[0] = np.nan
            py[0] = np.nan
            return (x > y) & (px <= py)

        def cross_down(x, y):
            px = np.roll(x, 1)
            py = np.roll(y, 1)
            px[0] = np.nan
            py[0] = np.nan
            return (x < y) & (px >= py)

        entry_long_mask = (
            cross_up(macd_vals, signal_vals)
            & (rsi_vals > 50)
            & (atr_vals > params["atr_threshold"])
        )
        entry_short_mask = (
            cross_down(macd_vals, signal_vals)
            & (rsi_vals < 50)
            & (atr_vals > params["atr_threshold"])
        )

        exit_mask = (hist_vals < 0) | (rsi_vals > 80) | (rsi_vals < 20)

        signals[entry_long_mask] = 1.0
        signals[entry_short_mask] = -1.0
        signals[exit_mask] = 0.0

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        long_entry = signals == 1.0
        short_entry = signals == -1.0

        df.loc[long_entry, "bb_stop_long"] = (
            close_vals[long_entry] - params["stop_atr_mult"] * atr_vals[long_entry]
        )
        df.loc[long_entry, "bb_tp_long"] = (
            close_vals[long_entry] + params["tp_atr_mult"] * atr_vals[long_entry]
        )
        df.loc[short_entry, "bb_stop_short"] = (
            close_vals[short_entry] + params["stop_atr_mult"] * atr_vals[short_entry]
        )
        df.loc[short_entry, "bb_tp_short"] = (
            close_vals[short_entry] - params["tp_atr_mult"] * atr_vals[short_entry]
        )
        signals.iloc[:warmup] = 0.0
        return signals
