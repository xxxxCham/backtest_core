from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="macd_adx_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["macd", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"leverage": 1, "stop_atr_mult": 2.0, "tp_atr_mult": 3.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "macd_fast_period": ParameterSpec(
                name="macd_fast_period",
                min_val=5,
                max_val=30,
                default=12,
                param_type="int",
                step=1,
            ),
            "macd_slow_period": ParameterSpec(
                name="macd_slow_period",
                min_val=10,
                max_val=50,
                default=26,
                param_type="int",
                step=1,
            ),
            "macd_signal_period": ParameterSpec(
                name="macd_signal_period",
                min_val=3,
                max_val=15,
                default=9,
                param_type="int",
                step=1,
            ),
            "adx_period": ParameterSpec(
                name="adx_period",
                min_val=5,
                max_val=30,
                default=14,
                param_type="int",
                step=1,
            ),
            "atr_period": ParameterSpec(
                name="atr_period",
                min_val=5,
                max_val=30,
                default=14,
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
                default=3.5,
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

        # Unpack indicator arrays
        macd_dict = indicators['macd']
        macd = np.nan_to_num(macd_dict["macd"])
        signal = np.nan_to_num(macd_dict["signal"])
        macd_hist = np.nan_to_num(macd_dict["histogram"])

        adx_val = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Helper cross functions
        def cross_up(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            px = np.roll(x, 1)
            py = np.roll(y, 1)
            px[0] = np.nan
            py[0] = np.nan
            return (x > y) & (px <= py)

        def cross_down(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            # Allow y to be a scalar
            if np.isscalar(y):
                y_arr = np.full_like(x, y, dtype=float)
            else:
                y_arr = y
            px = np.roll(x, 1)
            py = np.roll(y_arr, 1)
            px[0] = np.nan
            py[0] = np.nan
            return (x < y_arr) & (px >= py)

        # Entry conditions
        long_mask = cross_up(macd, signal) & (adx_val > 25)
        short_mask = cross_down(macd, signal) & (adx_val > 25)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        exit_mask = cross_down(macd_hist, 0.0) | (adx_val < 20)
        signals[exit_mask] = 0.0

        # Warmup period
        signals.iloc[:warmup] = 0.0

        # Risk management columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = float(params.get("stop_atr_mult", 2.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.5))

        long_entry = signals == 1.0
        short_entry = signals == -1.0

        df.loc[long_entry, "bb_stop_long"] = close[long_entry] - stop_atr_mult * atr[long_entry]
        df.loc[long_entry, "bb_tp_long"] = close[long_entry] + tp_atr_mult * atr[long_entry]

        df.loc[short_entry, "bb_stop_short"] = close[short_entry] + stop_atr_mult * atr[short_entry]
        df.loc[short_entry, "bb_tp_short"] = close[short_entry] - tp_atr_mult * atr[short_entry]

        return signals