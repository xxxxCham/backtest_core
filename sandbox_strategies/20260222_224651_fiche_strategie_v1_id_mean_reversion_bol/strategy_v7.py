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
            "leverage": 1,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
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
                max_val=4.0,
                default=1.5,
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
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=2.0,
                max_val=4.5,
                default=3.0,
                param_type="float",
                step=0.1,
            ),
        }

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # warm‑up period
        warmup = int(params.get("warmup", 50))

        # ----- indicator extraction -----
        close = np.nan_to_num(df["close"].values)

        bb = indicators['bollinger']
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])
        lower = np.nan_to_num(bb["lower"])

        rsi = np.nan_to_num(indicators['rsi'])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])

        # ----- helper for cross detection -----
        def cross_any(x: np.ndarray, y: np.ndarray | float) -> np.ndarray:
            """Return mask where x crosses y (either direction). Handles scalar y."""
            if np.isscalar(y):
                y_arr = np.full_like(x, y, dtype=float)
            else:
                y_arr = y

            prev_x = np.roll(x, 1)
            prev_y = np.roll(y_arr, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (
                (x > y_arr) & (prev_x <= prev_y)
            ) | ((x < y_arr) & (prev_x >= prev_y))

        # ----- entry masks -----
        long_mask = (
            (close < lower)
            & (rsi < params.get("rsi_oversold", 30))
            & (adx_val < 25)
        )
        short_mask = (
            (close > upper)
            & (rsi > params.get("rsi_overbought", 70))
            & (adx_val < 25)
        )

        # ----- exit mask -----
        exit_mask = cross_any(close, middle) | cross_any(rsi, 50)

        # ----- apply warm‑up -----
        signals.iloc[:warmup] = 0.0

        # ----- set exit signals -----
        signals[exit_mask] = 0.0

        # ----- set entry signals -----
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ----- risk management columns -----
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

        # ensure warm‑up still zeroed after risk columns added
        signals.iloc[:warmup] = 0.0

        return signals