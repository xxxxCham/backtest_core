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
            "rsi_period": 10,
            "stop_atr_mult": 2.0,
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
                default=10,
                param_type="int",
                step=1,
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                min_val=0.5,
                max_val=4.0,
                default=2.0,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=1.0,
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
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get("warmup", 50))

        # Indicator arrays
        close = df["close"].values
        bb = indicators['bollinger']
        lower = np.nan_to_num(bb["lower"])
        middle = np.nan_to_num(bb["middle"])
        upper = np.nan_to_num(bb["upper"])
        rsi = np.nan_to_num(indicators['rsi'])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])

        # Helper for cross detection that accepts scalars
        def cross_any(x: np.ndarray, y: np.ndarray | float) -> np.ndarray:
            if np.isscalar(y):
                y_arr = np.full_like(x, y)
            else:
                y_arr = y
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y_arr, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (
                (x > y_arr) & (prev_x <= prev_y)
            ) | ((x < y_arr) & (prev_x >= prev_y))

        # Long / short entry masks
        long_mask = (close < lower) & (rsi < params["rsi_oversold"]) & (adx_val < 25)
        short_mask = (close > upper) & (rsi > params["rsi_overbought"]) & (adx_val < 25)

        # Exit masks
        exit_cross_close = cross_any(close, middle)
        exit_cross_rsi = cross_any(rsi, 50)
        exit_mask = exit_cross_close | exit_cross_rsi

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # Warmup period
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Long entry SL/TP
        entry_long = signals == 1.0
        df.loc[entry_long, "bb_stop_long"] = (
            close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
        )
        df.loc[entry_long, "bb_tp_long"] = (
            close[entry_long] + params["tp_atr_mult"] * atr[entry_long]
        )

        # Short entry SL/TP
        entry_short = signals == -1.0
        df.loc[entry_short, "bb_stop_short"] = (
            close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
        )
        df.loc[entry_short, "bb_tp_short"] = (
            close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        )

        # Final warmup enforcement
        signals.iloc[:warmup] = 0.0
        return signals