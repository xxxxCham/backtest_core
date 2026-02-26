from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="trend_supertrend")

    @property
    def required_indicators(self) -> List[str]:
        return ["supertrend", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"leverage": 1, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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
                max_val=10.0,
                default=5.0,
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
        st = indicators['supertrend']
        direction = st["direction"].astype(float)  # keep NaNs for change detection
        adx_arr = indicators['adx']["adx"].astype(float)
        atr_arr = indicators['atr'].astype(float)

        # Entry masks
        long_mask = (direction == 1) & (adx_arr > 30)
        short_mask = (direction == -1) & (adx_arr > 30)

        # Exit mask: direction change or weak trend
        prev_dir = np.roll(direction, 1).astype(float)
        prev_dir[0] = np.nan
        direction_change = (
            (direction != prev_dir) & (~np.isnan(prev_dir)) & (~np.isnan(direction))
        )
        exit_mask = direction_change | (adx_arr < 25)

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Risk management columns
        df["bb_stop_long"] = np.nan
        df["bb_tp_long"] = np.nan
        df["bb_stop_short"] = np.nan
        df["bb_tp_short"] = np.nan

        stop_mult = params.get("stop_atr_mult", 2.0)
        tp_mult = params.get("tp_atr_mult", 5.0)

        close = df["close"].values
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr_arr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr_arr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr_arr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr_arr[short_mask]

        return signals