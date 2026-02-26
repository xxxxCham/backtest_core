from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mean_reversion_bollinger_rsi_adx_filter")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "rsi", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "adx_period": 14,
            "leverage": 1,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 13,
            "stop_atr_mult": 2.25,
            "tp_atr_mult": 6.0,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(
                name="rsi_period", min_val=5, max_val=50, default=13, param_type="int", step=1
            ),
            "adx_period": ParameterSpec(
                name="adx_period", min_val=5, max_val=30, default=14, param_type="int", step=1
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult", min_val=0.5, max_val=4.0, default=2.25, param_type="float", step=0.1
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult", min_val=1.0, max_val=10.0, default=6.0, param_type="float", step=0.1
            ),
            "leverage": ParameterSpec(
                name="leverage", min_val=1, max_val=2, default=1, param_type="int", step=1
            ),
        }

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        n = len(df)
        warmup = int(params.get("warmup", 50))

        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Ensure indicator arrays are numpy arrays
        bb_lower = np.nan_to_num(indicators['bollinger']["lower"])
        bb_middle = np.nan_to_num(indicators['bollinger']["middle"])
        bb_upper = np.nan_to_num(indicators['bollinger']["upper"])

        rsi_arr = np.nan_to_num(indicators['rsi'])
        adx_arr = np.nan_to_num(indicators['adx']["adx"])
        atr_arr = np.nan_to_num(indicators['atr'])

        close_arr = df["close"].values

        # Helper for detecting a cross between two arrays
        def cross_any(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            x = np.asarray(x)
            y = np.asarray(y)
            if y.ndim == 0:
                y = np.full_like(x, y)
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return ((x > y) & (prev_x <= prev_y)) | ((x < y) & (prev_x >= prev_y))

        # Entry conditions
        long_mask = (
            (close_arr < bb_lower)
            & (rsi_arr < params["rsi_oversold"])
            & (adx_arr < 20)
        )
        short_mask = (
            (close_arr > bb_upper)
            & (rsi_arr > params["rsi_overbought"])
            & (adx_arr < 20)
        )

        # Exit conditions
        rsi_cross_50 = (
            (rsi_arr > 50) & (np.roll(rsi_arr, 1) <= 50)
        ) | (
            (rsi_arr < 50) & (np.roll(rsi_arr, 1) >= 50)
        )
        exit_mask = (
            cross_any(close_arr, bb_middle)
            | rsi_cross_50
            | (adx_arr > 25)
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # ATR based stop‑loss and take‑profit
        df["bb_stop_long"] = np.nan
        df["bb_tp_long"] = np.nan
        df["bb_stop_short"] = np.nan
        df["bb_tp_short"] = np.nan

        df.loc[long_mask, "bb_stop_long"] = (
            close_arr[long_mask] - params["stop_atr_mult"] * atr_arr[long_mask]
        )
        df.loc[long_mask, "bb_tp_long"] = (
            close_arr[long_mask] + params["tp_atr_mult"] * atr_arr[long_mask]
        )
        df.loc[short_mask, "bb_stop_short"] = (
            close_arr[short_mask] + params["stop_atr_mult"] * atr_arr[short_mask]
        )
        df.loc[short_mask, "bb_tp_short"] = (
            close_arr[short_mask] - params["tp_atr_mult"] * atr_arr[short_mask]
        )

        # Ensure warm‑up period has no signals
        signals.iloc[:warmup] = 0.0

        return signals