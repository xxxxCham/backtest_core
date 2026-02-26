from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_bollinger_rsi_trend_filter')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'rsi', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'adx_threshold': 25,
            'leverage': 1,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'rsi_period': 11,
            'stop_atr_mult': 1.0,
            'tp_atr_mult': 3.5,
            'warmup': 20,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=11,
                param_type='int',
                step=1,
            ),
            'adx_threshold': ParameterSpec(
                name='adx_threshold',
                min_val=10,
                max_val=50,
                default=25,
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
                min_val=1.0,
                max_val=5.0,
                default=3.5,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=10,
                max_val=50,
                default=20,
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

        # Extract indicator arrays safely
        bb = indicators['bollinger']
        lower = np.nan_to_num(bb["lower"])
        middle = np.nan_to_num(bb["middle"])
        upper = np.nan_to_num(bb["upper"])
        rsi = np.nan_to_num(indicators['rsi'])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Helper for cross_any that accepts a scalar threshold
        def cross_any(x: np.ndarray, y: np.ndarray | float | int) -> np.ndarray:
            if np.isscalar(y):
                y = np.full_like(x, y, dtype=np.float64)
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return ((x > y) & (prev_x <= prev_y)) | ((x < y) & (prev_x >= prev_y))

        # Entry masks with trend filter
        long_mask = (
            (close < lower)
            & (rsi < params["rsi_oversold"])
            & (adx_val < params["adx_threshold"])
        )
        short_mask = (
            (close > upper)
            & (rsi > params["rsi_overbought"])
            & (adx_val < params["adx_threshold"])
        )

        # Exit mask
        exit_mask = cross_any(close, middle) | cross_any(rsi, 50.0)

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Prepare SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Long entry SL/TP
        entry_long = signals == 1.0
        df.loc[entry_long, "bb_stop_long"] = (
            close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
        )
        df.loc[entry_long, "bb_tp_long"] = (
            close[entry_long] + params["tp_atr_mult"] * atr[entry_long]
        )

        # Short entry SL/TP
        entry_short = signals == -1.0
        df.loc[entry_short, "bb_stop_short"] = (
            close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
        )
        df.loc[entry_short, "bb_tp_short"] = (
            close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        )
        signals.iloc[:warmup] = 0.0
        return signals