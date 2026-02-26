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
        return {'leverage': 1,
         'rsi_overbought': 65,
         'rsi_oversold': 40,
         'rsi_period': 13,
         'stop_atr_mult': 1.2,
         'tp_atr_mult': 3.5,
         'warmup': 50}
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
                default=1.0,
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
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        warmup = int(params.get("warmup", 50))

        # Extract indicator arrays
        bb = indicators['bollinger']
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])
        lower = np.nan_to_num(bb["lower"])

        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])

        close = df["close"].values

        # Helper for cross detection
        def cross_any(x: np.ndarray, y: np.ndarray | float) -> np.ndarray:
            # Ensure y is an array of the same length
            if np.isscalar(y):
                y = np.full_like(x, y, dtype=float)
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (
                ((x > y) & (prev_x <= prev_y))
                | ((x < y) & (prev_x >= prev_y))
            )

        # Entry conditions
        long_mask = (close < lower) & (rsi < params["rsi_oversold"])
        short_mask = (close > upper) & (rsi > params["rsi_overbought"])

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        exit_mask = cross_any(close, middle) | cross_any(rsi, 50.0)
        signals[exit_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]

        long_entry_mask = signals == 1.0
        short_entry_mask = signals == -1.0

        df.loc[long_entry_mask, "bb_stop_long"] = (
            close[long_entry_mask] - stop_atr_mult * atr[long_entry_mask]
        )
        df.loc[long_entry_mask, "bb_tp_long"] = (
            close[long_entry_mask] + tp_atr_mult * atr[long_entry_mask]
        )

        df.loc[short_entry_mask, "bb_stop_short"] = (
            close[short_entry_mask] + stop_atr_mult * atr[short_entry_mask]
        )
        df.loc[short_entry_mask, "bb_tp_short"] = (
            close[short_entry_mask] - tp_atr_mult * atr[short_entry_mask]
        )

        signals.iloc[:warmup] = 0.0
        return signals