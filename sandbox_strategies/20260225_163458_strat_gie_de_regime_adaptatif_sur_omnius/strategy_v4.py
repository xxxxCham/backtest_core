from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="vwap_rsi_atr_regime_adapt")

    @property
    def required_indicators(self) -> List[str]:
        return ["vwap", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "leverage": 1,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 1.1,
            "tp_atr_mult": 2.5,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(
                name="rsi_period",
                min_val=5,
                max_val=50,
                default=14,
                param_type="int",
                step=1,
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                min_val=0.5,
                max_val=3.0,
                default=1.1,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=1.5,
                max_val=5.0,
                default=2.5,
                param_type="float",
                step=0.1,
            ),
            "leverage": ParameterSpec(
                name="leverage",
                min_val=1,
                max_val=2,
                default=1,
                param_type="int",
                step=1,
            ),
        }

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        n = len(df)
        warmup = int(params.get("warmup", 50))
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # unwrap indicators and ensure they are arrays of length n
        close = df["close"].values
        vwap_raw = indicators['vwap']
        rsi_raw = indicators['rsi']
        atr_raw = indicators['atr']

        vwap = np.nan_to_num(vwap_raw)
        rsi = np.nan_to_num(rsi_raw)
        atr = np.nan_to_num(atr_raw)

        # broadcast scalar indicators to full length if necessary
        if vwap.ndim == 0:
            vwap = np.full(n, vwap)
        if rsi.ndim == 0:
            rsi = np.full(n, rsi)
        if atr.ndim == 0:
            atr = np.full(n, atr)

        # helper cross functions that handle scalar thresholds
        def cross_up(x: np.ndarray, y: np.ndarray | float) -> np.ndarray:
            prev_x = np.roll(x, 1)
            prev_x[0] = np.nan
            if np.isscalar(y):
                prev_y = np.full_like(x, y)
            else:
                prev_y = np.roll(y, 1)
                prev_y[0] = np.nan
            return (x > y) & (prev_x <= prev_y)

        def cross_down(x: np.ndarray, y: np.ndarray | float) -> np.ndarray:
            prev_x = np.roll(x, 1)
            prev_x[0] = np.nan
            if np.isscalar(y):
                prev_y = np.full_like(x, y)
            else:
                prev_y = np.roll(y, 1)
                prev_y[0] = np.nan
            return (x < y) & (prev_x >= prev_y)

        # Entry conditions
        long_mask = (close > vwap) & (rsi > 50)
        short_mask = (close < vwap) & (rsi < 50)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        exit_mask = (
            cross_up(close, vwap)
            | cross_down(close, vwap)
            | cross_up(rsi, 50.0)
            | cross_down(rsi, 50.0)
        )
        signals[exit_mask] = 0.0

        # warmup protection
        signals.iloc[:warmup] = 0.0

        # initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params.get("stop_atr_mult", 1.1)
        tp_atr_mult = params.get("tp_atr_mult", 2.5)

        # compute SL/TP for long entries
        entry_long_mask = signals == 1.0
        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_atr_mult * atr[
            entry_long_mask
        ]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_atr_mult * atr[
            entry_long_mask
        ]

        # compute SL/TP for short entries
        entry_short_mask = signals == -1.0
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_atr_mult * atr[
            entry_short_mask
        ]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_atr_mult * atr[
            entry_short_mask
        ]

        signals.iloc[:warmup] = 0.0
        return signals