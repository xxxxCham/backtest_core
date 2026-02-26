from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mean_reversion_bollinger_rsi_optimized")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"leverage": 1, "stop_atr_mult": 1.0, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                min_val=0.5,
                max_val=3.0,
                default=1.0,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=1.0,
                max_val=5.0,
                default=2.0,
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

        # Initialize signals
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Extract indicator arrays
        bb = indicators['bollinger']
        rsi_arr = np.nan_to_num(indicators['rsi'])
        atr_arr = np.nan_to_num(indicators['atr'])
        close_arr = df["close"].values

        lower = np.nan_to_num(bb["lower"])
        middle = np.nan_to_num(bb["middle"])
        upper = np.nan_to_num(bb["upper"])

        # Helper to detect cross between two series
        def cross_any(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            # Ensure y is an array of the same shape
            y_arr = np.full_like(x, y) if np.isscalar(y) else y
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y_arr, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return ((x > y_arr) & (prev_x <= prev_y)) | ((x < y_arr) & (prev_x >= prev_y))

        # Entry masks
        long_mask = (close_arr < lower) & (rsi_arr < 30)
        short_mask = (close_arr > upper) & (rsi_arr > 70)

        # Exit mask
        exit_mask = cross_any(close_arr, middle) | cross_any(rsi_arr, 50)

        # Apply warmup: ignore signals in the first `warmup` rows
        long_mask[:warmup] = False
        short_mask[:warmup] = False
        exit_mask[:warmup] = False

        # Set signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # ATR-based stop‑loss and take‑profit
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = float(params.get("stop_atr_mult", 1.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.0))

        long_entries = signals == 1.0
        short_entries = signals == -1.0

        df.loc[long_entries, "bb_stop_long"] = close_arr[long_entries] - stop_atr_mult * atr_arr[long_entries]
        df.loc[long_entries, "bb_tp_long"] = close_arr[long_entries] + tp_atr_mult * atr_arr[long_entries]

        df.loc[short_entries, "bb_stop_short"] = close_arr[short_entries] + stop_atr_mult * atr_arr[short_entries]
        df.loc[short_entries, "bb_tp_short"] = close_arr[short_entries] - tp_atr_mult * atr_arr[short_entries]

        return signals