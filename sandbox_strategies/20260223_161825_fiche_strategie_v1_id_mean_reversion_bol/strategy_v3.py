from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mean_reversion_bollinger_rsi_adx")

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
            "adx_threshold": 25,  # added to avoid KeyError
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
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=1.0,
                max_val=5.0,
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
            # adx_threshold is optional for user tuning but not exposed here
        }

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)

        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Prepare indicator arrays
        close = df["close"].values
        boll = indicators['bollinger']
        lower = np.nan_to_num(boll["lower"])
        middle = np.nan_to_num(boll["middle"])
        upper = np.nan_to_num(boll["upper"])
        rsi = np.nan_to_num(indicators['rsi'])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])

        # Helper for cross detection
        def cross_any(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return ((x > y) & (prev_x <= prev_y)) | ((x < y) & (prev_x >= prev_y))

        # Entry conditions
        long_mask = (
            (close < lower)
            & (rsi < params["rsi_oversold"])
            & (adx_val < params.get("adx_threshold", 25))
        )
        short_mask = (
            (close > upper)
            & (rsi > params["rsi_overbought"])
            & (adx_val < params.get("adx_threshold", 25))
        )

        # Exit conditions
        rsi_cross = (
            (rsi > 50) & (np.roll(rsi, 1) <= 50)
            | (rsi < 50) & (np.roll(rsi, 1) >= 50)
        )
        exit_mask = cross_any(close, middle) | rsi_cross

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # ATR-based SL/TP for longs
        long_entries = signals == 1.0
        df.loc[long_entries, "bb_stop_long"] = (
            close[long_entries] - params["stop_atr_mult"] * atr[long_entries]
        )
        df.loc[long_entries, "bb_tp_long"] = (
            close[long_entries] + params["tp_atr_mult"] * atr[long_entries]
        )

        # ATR-based SL/TP for shorts
        short_entries = signals == -1.0
        df.loc[short_entries, "bb_stop_short"] = (
            close[short_entries] + params["stop_atr_mult"] * atr[short_entries]
        )
        df.loc[short_entries, "bb_tp_short"] = (
            close[short_entries] - params["tp_atr_mult"] * atr[short_entries]
        )

        signals.iloc[:warmup] = 0.0
        return signals