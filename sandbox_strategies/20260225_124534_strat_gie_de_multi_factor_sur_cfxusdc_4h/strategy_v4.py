from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='supertrend_stoch_atr_multi_factor')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'stochastic', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'atr_period': 14,
            'atr_threshold': 0.005,
            'leverage': 1,
            'stochastic_d_period': 3,
            'stochastic_k_period': 14,
            'stochastic_smooth_k': 3,
            'stop_atr_mult': 2.2,
            'supertrend_multiplier': 3.0,
            'supertrend_period': 10,
            'tp_atr_mult': 5.5,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'supertrend_period': ParameterSpec(
                name='supertrend_period',
                min_val=5,
                max_val=30,
                default=10,
                param_type='int',
                step=1,
            ),
            'supertrend_multiplier': ParameterSpec(
                name='supertrend_multiplier',
                min_val=1.0,
                max_val=5.0,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
            'stochastic_k_period': ParameterSpec(
                name='stochastic_k_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stochastic_d_period': ParameterSpec(
                name='stochastic_d_period',
                min_val=2,
                max_val=10,
                default=3,
                param_type='int',
                step=1,
            ),
            'stochastic_smooth_k': ParameterSpec(
                name='stochastic_smooth_k',
                min_val=1,
                max_val=5,
                default=3,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=7,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_threshold': ParameterSpec(
                name='atr_threshold',
                min_val=0.001,
                max_val=0.02,
                default=0.005,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=2.2,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.5,
                max_val=10.0,
                default=5.5,
                param_type='float',
                step=0.1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=5,
                default=1,
                param_type='int',
                step=1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=20,
                max_val=200,
                default=50,
                param_type='int',
                step=1,
            ),
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)

        # Warm‑up protection
        warmup = int(params.get('warmup', 50))
        signals.iloc[:warmup] = 0.0

        # Prepare boolean masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # ---- Indicator extraction ------------------------------------------------
        st = indicators['supertrend']
        direction = np.nan_to_num(st["direction"])

        stoch = indicators['stochastic']
        k = np.nan_to_num(stoch["stoch_k"])
        d = np.nan_to_num(stoch["stoch_d"])

        atr = np.nan_to_num(indicators['atr'])

        # ---- Parameters ------------------------------------------------------------
        atr_threshold = float(params.get("atr_threshold", 0.005))
        stop_atr_mult = float(params.get("stop_atr_mult", 2.2))
        tp_atr_mult = float(params.get("tp_atr_mult", 5.5))

        # ---- Stochastic cross detection --------------------------------------------
        prev_k = np.roll(k, 1)
        prev_d = np.roll(d, 1)
        prev_k[0] = np.nan
        prev_d[0] = np.nan
        cross_up = (k > d) & (prev_k <= prev_d)
        cross_down = (k < d) & (prev_k >= prev_d)

        # ---- Entry conditions ------------------------------------------------------
        long_entry = (direction == 1) & (k > d) & (k > 20) & (atr > atr_threshold)
        short_entry = (direction == -1) & (k < d) & (k < 80) & (atr > atr_threshold)

        long_mask[long_entry] = True
        short_mask[short_entry] = True

        # ---- Set entry signals ------------------------------------------------------
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ---- Exit conditions --------------------------------------------------------
        exit_long = (direction == -1) | cross_down | (atr < atr_threshold)
        exit_short = (direction == 1) | cross_up | (atr < atr_threshold)

        signals[exit_long] = 0.0
        signals[exit_short] = 0.0

        # ---- Prepare SL/TP columns -------------------------------------------------
        df["bb_stop_long"] = np.nan
        df["bb_tp_long"] = np.nan
        df["bb_stop_short"] = np.nan
        df["bb_tp_short"] = np.nan

        close = df["close"].values

        # Column positions for fast .iloc assignment
        col_stop_long = df.columns.get_loc("bb_stop_long")
        col_tp_long = df.columns.get_loc("bb_tp_long")
        col_stop_short = df.columns.get_loc("bb_stop_short")
        col_tp_short = df.columns.get_loc("bb_tp_short")

        # ---- Write ATR‑based stop‑loss / take‑profit for longs ----------------------
        long_idx = np.where(long_mask)[0]
        if long_idx.size:
            df.iloc[long_idx, col_stop_long] = close[long_idx] - stop_atr_mult * atr[long_idx]
            df.iloc[long_idx, col_tp_long] = close[long_idx] + tp_atr_mult * atr[long_idx]

        # ---- Write ATR‑based stop‑loss / take‑profit for shorts ---------------------
        short_idx = np.where(short_mask)[0]
        if short_idx.size:
            df.iloc[short_idx, col_stop_short] = close[short_idx] + stop_atr_mult * atr[short_idx]
            df.iloc[short_idx, col_tp_short] = close[short_idx] - tp_atr_mult * atr[short_idx]

        # Ensure warm‑up period remains flat
        signals.iloc[:warmup] = 0.0
        return signals