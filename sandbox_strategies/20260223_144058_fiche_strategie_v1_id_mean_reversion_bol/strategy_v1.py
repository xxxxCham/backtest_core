from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mean_reversion_bollinger_rsi")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "leverage": 1,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 13,
            "stop_atr_mult": 1.0,
            "tp_atr_mult": 4.0,
            "warmup": 23,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(
                name="rsi_period",
                min_val=5,
                max_val=50,
                default=13,
                param_type="int",
                step=1,
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                min_val=0.5,
                max_val=4.0,
                default=1.0,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=1.0,
                max_val=10.0,
                default=4.0,
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

        # Extract indicator arrays
        close = df["close"].values
        bb = indicators['bollinger']
        lower = np.nan_to_num(bb["lower"])
        middle = np.nan_to_num(bb["middle"])
        upper = np.nan_to_num(bb["upper"])
        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])

        # Helper to detect a cross of two series or a series with a constant
        def cross_any(x: np.ndarray, y: np.ndarray | float) -> np.ndarray:
            prev_x = np.roll(x, 1)
            prev_x[0] = np.nan
            if np.ndim(y) == 0:  # scalar
                return (
                    ((x > y) & (prev_x <= y))
                    | ((x < y) & (prev_x >= y))
                )
            else:
                prev_y = np.roll(y, 1)
                prev_y[0] = np.nan
                return (
                    ((x > y) & (prev_x <= prev_y))
                    | ((x < y) & (prev_x >= prev_y))
                )

        # Entry signals
        long_mask = (close < lower) & (rsi < params["rsi_oversold"])
        short_mask = (close > upper) & (rsi > params["rsi_overbought"])
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit signals
        exit_mask = cross_any(close, middle) | cross_any(rsi, 50.0)
        signals[exit_mask] = 0.0

        # Warm‑up period
        signals.iloc[:warmup] = 0.0

        # ATR‑based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = signals == 1.0
        entry_short = signals == -1.0

        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - (
            params["stop_atr_mult"] * atr[entry_long]
        )
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + (
            params["tp_atr_mult"] * atr[entry_long]
        )
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + (
            params["stop_atr_mult"] * atr[entry_short]
        )
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - (
            params["tp_atr_mult"] * atr[entry_short]
        )

        return signals