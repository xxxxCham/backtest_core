from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="supertrend_macd_adx")

    @property
    def required_indicators(self) -> List[str]:
        return ["supertrend", "macd", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"leverage": 1, "stop_atr_mult": 2.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                min_val=0.5,
                max_val=4.0,
                default=2.5,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=1.0,
                max_val=6.0,
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
        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])
        direction = np.nan_to_num(indicators['supertrend']["direction"]).astype(float)
        adx_val = np.nan_to_num(indicators['adx']["adx"]).astype(float)
        macd_hist = np.nan_to_num(indicators['macd']["histogram"]).astype(float)

        # Entry conditions
        long_mask = (direction == 1) & (macd_hist > 0) & (adx_val > 25)
        short_mask = (direction == -1) & (macd_hist < 0) & (adx_val > 25)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        # Compute direction change without NaN assignment
        dir_change = np.concatenate([[False], direction[1:] != direction[:-1]])

        # MACD histogram cross
        prev_hist = np.concatenate([[np.nan], macd_hist[:-1]])
        cross_up = (macd_hist > 0) & (prev_hist <= 0)
        cross_down = (macd_hist < 0) & (prev_hist >= 0)
        hist_cross = cross_up | cross_down

        adx_below_20 = adx_val < 20

        exit_mask = dir_change | hist_cross | adx_below_20
        signals[exit_mask] = 0.0

        # Warmup protection
        signals[:warmup] = 0.0

        # ATR-based stop and take‑profit levels on entry bars
        stop_mult = float(params.get("stop_atr_mult", 2.5))
        tp_mult = float(params.get("tp_atr_mult", 3.0))

        # Long entries
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]
        # Short entries
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]

        return signals