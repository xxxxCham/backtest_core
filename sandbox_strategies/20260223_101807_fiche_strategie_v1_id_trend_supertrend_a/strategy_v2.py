from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='trend_supertrend_rsi_filter')

    @property
    def required_indicators(self) -> List[str]:
        # ATR is needed for risk management
        return ['supertrend', 'adx', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'leverage': 1,
            'rsi_period': 14,
            'stop_atr_mult': 1.5,
            'tp_atr_mult': 2.5,
            'warmup': 50,
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
                max_val=5.0,
                default=2.5,
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
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        n = len(df)
        warmup = int(params.get('warmup', 50))

        # Boolean masks for long/short entries
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Extract indicator arrays safely
        st_dir = np.nan_to_num(indicators['supertrend']["direction"])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        rsi_val = np.nan_to_num(indicators['rsi'])
        atr_arr = np.nan_to_num(indicators['atr'])
        close_arr = df["close"].values

        # Entry conditions
        long_mask = (st_dir == 1) & (adx_val > 30) & (rsi_val > 60)
        short_mask = (st_dir == -1) & (adx_val > 30) & (rsi_val < 40)

        # Initialize signals series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR‑based stop‑loss and take‑profit levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_mult = float(params.get("stop_atr_mult", 1.5))
        tp_mult = float(params.get("tp_atr_mult", 2.5))

        entry_long = signals == 1.0
        entry_short = signals == -1.0

        df.loc[entry_long, "bb_stop_long"] = close_arr[entry_long] - stop_mult * atr_arr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close_arr[entry_long] + tp_mult * atr_arr[entry_long]

        df.loc[entry_short, "bb_stop_short"] = close_arr[entry_short] + stop_mult * atr_arr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close_arr[entry_short] - tp_mult * atr_arr[entry_short]

        return signals