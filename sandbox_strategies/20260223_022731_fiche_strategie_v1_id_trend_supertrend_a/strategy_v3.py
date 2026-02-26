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
        return ['supertrend', 'adx', 'atr', 'rsi']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_threshold_exit': 20,
         'adx_threshold_long': 30,
         'leverage': 1,
         'rsi_period': 14,
         'rsi_threshold_long': 55,
         'rsi_threshold_short': 45,
         'stop_atr_mult': 1.25,
         'tp_atr_mult': 4.0,
         'warmup': 50}

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
                default=1.25,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=4.0,
                param_type='float',
                step=0.1,
            ),
            'adx_threshold_long': ParameterSpec(
                name='adx_threshold_long',
                min_val=20,
                max_val=50,
                default=30,
                param_type='int',
                step=1,
            ),
            'adx_threshold_exit': ParameterSpec(
                name='adx_threshold_exit',
                min_val=10,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'rsi_threshold_long': ParameterSpec(
                name='rsi_threshold_long',
                min_val=50,
                max_val=70,
                default=55,
                param_type='int',
                step=1,
            ),
            'rsi_threshold_short': ParameterSpec(
                name='rsi_threshold_short',
                min_val=30,
                max_val=50,
                default=45,
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

        signals.iloc[:warmup] = 0.0

        # Extract indicator arrays
        st_dir = np.nan_to_num(indicators['supertrend']["direction"])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Long entry conditions
        long_cond = (
            (st_dir == 1)
            & (adx_val > params["adx_threshold_long"])
            & (rsi > params["rsi_threshold_long"])
        )
        long_mask = long_cond

        # Short entry conditions
        short_cond = (
            (st_dir == -1)
            & (adx_val > params["adx_threshold_long"])
            & (rsi < params["rsi_threshold_short"])
        )
        short_mask = short_cond

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        # Direction change detection
        prev_dir = np.roll(st_dir, 1)
        prev_dir[0] = 0
        dir_change = (st_dir != prev_dir) & (prev_dir != 0)

        # RSI cross 50
        rsi_prev = np.roll(rsi, 1)
        rsi_prev[0] = np.nan
        rsi_cross_up = (rsi > 50) & (rsi_prev <= 50)
        rsi_cross_down = (rsi < 50) & (rsi_prev >= 50)
        rsi_cross_any = rsi_cross_up | rsi_cross_down

        exit_mask = dir_change | (adx_val < params["adx_threshold_exit"]) | rsi_cross_any
        signals[exit_mask] = 0.0

        # ATR-based SL/TP levels on entry bars
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = long_mask
        entry_short = short_mask

        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]

        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
