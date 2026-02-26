from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_adx_mean_reversion")

    @property
    def required_indicators(self) -> List[str]:
        # ATR is needed for the risk module, ADX for filtering
        return ["bollinger", "rsi", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "leverage": 1,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 8,
            "stop_atr_mult": 2.0,
            "tp_atr_mult": 4.5,
            "warmup": 50,
            # ADX thresholds with defaults
            "adx_entry_threshold": 20,
            "adx_exit_threshold": 25,
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
                max_val=4.0,
                default=2.25,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=1.0,
                max_val=10.0,
                default=5.0,
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
            # ADX thresholds are optional parameters but kept for completeness
            "adx_entry_threshold": ParameterSpec(
                name="adx_entry_threshold",
                min_val=0,
                max_val=100,
                default=20,
                param_type="int",
                step=1,
            ),
            "adx_exit_threshold": ParameterSpec(
                name="adx_exit_threshold",
                min_val=0,
                max_val=100,
                default=25,
                param_type="int",
                step=1,
            ),
        }

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)

        # Warm‑up period
        warmup = int(params.get("warmup", 50))

        # Indicator arrays
        close = df["close"].values
        bb = indicators['bollinger']
        lower = np.nan_to_num(bb["lower"])
        middle = np.nan_to_num(bb["middle"])
        upper = np.nan_to_num(bb["upper"])
        rsi = np.nan_to_num(indicators['rsi'])
        adx = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])

        # Helper to detect a cross
        def cross_any(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return ((x > y) & (prev_x <= prev_y)) | ((x < y) & (prev_x >= prev_y))

        # Entry masks
        long_mask = (
            (close < lower)
            & (rsi < params["rsi_oversold"])
            & (adx < params["adx_entry_threshold"])
        )
        short_mask = (
            (close > upper)
            & (rsi > params["rsi_overbought"])
            & (adx < params["adx_entry_threshold"])
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit masks
        exit_cross_middle = cross_any(close, middle)
        exit_cross_rsi = cross_any(rsi, np.full(n, 50.0))
        exit_adx = adx > params["adx_exit_threshold"]
        exit_mask = exit_cross_middle | exit_cross_rsi | exit_adx

        signals[exit_mask] = 0.0

        # Apply warm‑up
        signals.iloc[:warmup] = 0.0

        # ATR‑based stop / take‑profit columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long_mask = signals == 1.0
        entry_short_mask = signals == -1.0

        df.loc[entry_long_mask, "bb_stop_long"] = (
            close[entry_long_mask] - params["stop_atr_mult"] * atr[entry_long_mask]
        )
        df.loc[entry_long_mask, "bb_tp_long"] = (
            close[entry_long_mask] + params["tp_atr_mult"] * atr[entry_long_mask]
        )

        df.loc[entry_short_mask, "bb_stop_short"] = (
            close[entry_short_mask] + params["stop_atr_mult"] * atr[entry_short_mask]
        )
        df.loc[entry_short_mask, "bb_tp_short"] = (
            close[entry_short_mask] - params["tp_atr_mult"] * atr[entry_short_mask]
        )

        return signals