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
        return {
            "adx_period": 22,
            "atr_period": 14,
            "leverage": 1,
            "stop_atr_mult": 2.75,
            "supertrend_atr_period": 13,
            "supertrend_multiplier": 3.0,
            "tp_atr_mult": 5.5,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "supertrend_atr_period": ParameterSpec(
                name="supertrend_atr_period",
                min_val=5,
                max_val=20,
                default=13,
                param_type="int",
                step=1,
            ),
            "supertrend_multiplier": ParameterSpec(
                name="supertrend_multiplier",
                min_val=1,
                max_val=5,
                default=3.0,
                param_type="float",
                step=0.1,
            ),
            "adx_period": ParameterSpec(
                name="adx_period",
                min_val=10,
                max_val=30,
                default=22,
                param_type="int",
                step=1,
            ),
            "atr_period": ParameterSpec(
                name="atr_period",
                min_val=5,
                max_val=20,
                default=14,
                param_type="int",
                step=1,
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                min_val=0.5,
                max_val=5.0,
                default=2.75,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=1.0,
                max_val=10.0,
                default=5.5,
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
        st = indicators['supertrend']
        # Ensure direction is float and NaNs become 0
        direction = st["direction"].astype(float)
        direction[np.isnan(direction)] = 0.0

        adx_d = indicators['adx']
        adx_val = adx_d["adx"].astype(float)
        adx_val[np.isnan(adx_val)] = 0.0

        atr_arr = indicators['atr'].astype(float)
        atr_arr[np.isnan(atr_arr)] = 0.0

        close = df["close"].values

        # Entry conditions
        long_mask = (direction == 1) & (adx_val > 25)
        short_mask = (direction == -1) & (adx_val > 25)

        # Exit conditions
        rolled_dir = np.roll(direction, 1)
        direction_change = (direction != rolled_dir) & (np.arange(n) > 0)
        adx_weak = adx_val < 20
        exit_mask = direction_change | adx_weak

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # Warmup period
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP levels on entry bars
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params.get("stop_atr_mult", 2.75)
        tp_atr_mult = params.get("tp_atr_mult", 5.5)

        # Long entries
        long_entries = signals == 1.0
        df.loc[long_entries, "bb_stop_long"] = (
            close[long_entries] - stop_atr_mult * atr_arr[long_entries]
        )
        df.loc[long_entries, "bb_tp_long"] = (
            close[long_entries] + tp_atr_mult * atr_arr[long_entries]
        )

        # Short entries
        short_entries = signals == -1.0
        df.loc[short_entries, "bb_stop_short"] = (
            close[short_entries] + stop_atr_mult * atr_arr[short_entries]
        )
        df.loc[short_entries, "bb_tp_short"] = (
            close[short_entries] - tp_atr_mult * atr_arr[short_entries]
        )

        # Re‑apply warmup to avoid any accidental signals
        signals.iloc[:warmup] = 0.0
        return signals