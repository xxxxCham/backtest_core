from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_bollinger_rsi_adx')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'rsi', 'atr', 'adx']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'leverage': 1,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'rsi_period': 14,
            'stop_atr_mult': 1.5,
            'tp_atr_mult': 3.5,
            'warmup': 36,
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
                default=3.5,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=20,
                max_val=50,
                default=36,
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

    def _broadcast_to_length(self, arr: np.ndarray, length: int) -> np.ndarray:
        """Ensure the array has the desired length, broadcasting scalars if needed."""
        if np.ndim(arr) == 0:
            return np.full(length, arr, dtype=float)
        if len(arr) != length:
            # Pad or truncate to match length
            if len(arr) < length:
                pad = np.full(length - len(arr), np.nan, dtype=float)
                return np.concatenate([arr, pad])
            return arr[:length]
        return arr

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        warmup = int(params.get('warmup', 50))
        signals.iloc[:warmup] = 0.0

        close = df["close"].values
        bb = indicators['bollinger']
        lower = self._broadcast_to_length(np.nan_to_num(bb["lower"]), n)
        middle = self._broadcast_to_length(np.nan_to_num(bb["middle"]), n)
        upper = self._broadcast_to_length(np.nan_to_num(bb["upper"]), n)

        rsi = self._broadcast_to_length(np.nan_to_num(indicators['rsi']), n)
        adx_val = self._broadcast_to_length(np.nan_to_num(indicators['adx']["adx"]), n)
        atr = self._broadcast_to_length(np.nan_to_num(indicators['atr']), n)

        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)

        long_mask = (close < lower) & (rsi < rsi_oversold) & (adx_val < 25)
        short_mask = (close > upper) & (rsi > rsi_overbought) & (adx_val < 25)

        def cross_any(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return ((x > y) & (prev_x <= prev_y)) | ((x < y) & (prev_x >= prev_y))

        exit_mask = (
            cross_any(close, middle)
            | cross_any(rsi, np.full(n, 50.0))
            | (adx_val > 25)
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # Risk management columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.5)

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        signals.iloc[:warmup] = 0.0
        return signals