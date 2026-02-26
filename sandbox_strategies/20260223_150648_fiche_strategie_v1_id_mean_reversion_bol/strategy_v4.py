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
        return ["bollinger", "rsi", "atr", "adx"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "leverage": 1,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 8,
            "stop_atr_mult": 2.25,
            "tp_atr_mult": 5.0,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(
                name="rsi_period",
                min_val=5,
                max_val=50,
                default=8,
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
                max_val=8.0,
                default=5.0,
                param_type="float",
                step=0.1,
            ),
            "warmup": ParameterSpec(
                name="warmup",
                min_val=20,
                max_val=100,
                default=50,
                param_type="int",
                step=1,
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

        # Helper to detect crossovers, handles scalar thresholds
        def cross_any(x: np.ndarray, y: np.ndarray | float) -> np.ndarray:
            x_arr = np.asarray(x)
            if np.isscalar(y):
                y_arr = np.full_like(x_arr, y, dtype=float)
            else:
                y_arr = np.asarray(y)
            prev_x = np.roll(x_arr, 1)
            prev_y = np.roll(y_arr, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (
                (x_arr > y_arr) & (prev_x <= prev_y)
            ) | ((x_arr < y_arr) & (prev_x >= prev_y))

        # Unwrap indicators
        close = df["close"].values
        bb = indicators['bollinger']
        lower = np.nan_to_num(bb["lower"])
        middle = np.nan_to_num(bb["middle"])
        upper = np.nan_to_num(bb["upper"])
        rsi = np.nan_to_num(indicators['rsi'])
        adx = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])

        # Entry masks
        long_mask = (
            (close < lower)
            & (rsi < params.get("rsi_oversold", 30))
            & (adx < 20)
        )
        short_mask = (
            (close > upper)
            & (rsi > params.get("rsi_overbought", 70))
            & (adx < 20)
        )

        # Exit mask
        exit_mask = (
            cross_any(close, middle)
            | cross_any(rsi, 50)
            | (adx > 25)
        )

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # Warmup period
        signals.iloc[:warmup] = 0.0

        # Risk columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = signals == 1.0
        entry_short = signals == -1.0

        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - (
            params.get("stop_atr_mult", 2.25) * atr[entry_long]
        )
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + (
            params.get("tp_atr_mult", 5.0) * atr[entry_long]
        )

        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + (
            params.get("stop_atr_mult", 2.25) * atr[entry_short]
        )
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - (
            params.get("tp_atr_mult", 5.0) * atr[entry_short]
        )

        signals.iloc[:warmup] = 0.0
        return signals