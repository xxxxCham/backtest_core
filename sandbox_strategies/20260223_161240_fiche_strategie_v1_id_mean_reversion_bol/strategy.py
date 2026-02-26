from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mean_reversion_bollinger_rsi_atr_filter")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "atr_threshold": 1.5,
            "leverage": 1,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 2.0,
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
            "atr_threshold": ParameterSpec(
                name="atr_threshold",
                min_val=0.5,
                max_val=3.0,
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
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                min_val=1.0,
                max_val=2.0,
                default=2.0,
                param_type="float",
                step=0.1,
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
        warmup = int(params.get("warmup", 50))

        # Initialise signal series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Extract indicator arrays
        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])
        rsi = np.nan_to_num(indicators['rsi'])

        bb = indicators['bollinger']
        lower = np.nan_to_num(bb["lower"])
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])

        # Parameters
        atr_threshold = params["atr_threshold"]
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]

        # Long/short entry masks
        long_mask = (close < lower) & (rsi < rsi_oversold) & (atr > atr_threshold)
        short_mask = (close > upper) & (rsi > rsi_overbought) & (atr > atr_threshold)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Helper for cross‑overs
        def cross_any(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            # Ensure y is an array of the same length as x
            if np.isscalar(y):
                y = np.full_like(x, y)
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return ((x > y) & (prev_x <= prev_y)) | ((x < y) & (prev_x >= prev_y))

        # Exit mask
        exit_mask = cross_any(close, middle) | cross_any(rsi, 50.0)
        signals[exit_mask] = 0.0

        # ATR‑based stop/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        # Ensure warmup period is flat
        signals.iloc[:warmup] = 0.0

        return signals