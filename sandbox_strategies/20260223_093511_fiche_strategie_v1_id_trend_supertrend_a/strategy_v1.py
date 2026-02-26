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
            "atr_period": 14,
            "leverage": 1,
            "stop_atr_mult": 1.75,
            "tp_atr_mult": 4.0,
            "warmup": 30,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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
                default=1.75,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=1.0,
                max_val=10.0,
                default=4.0,
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

        # Extract indicator arrays
        direction = np.nan_to_num(indicators['supertrend']["direction"])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Entry conditions
        long_mask = (direction == 1) & (adx_val > 30)
        short_mask = (direction == -1) & (adx_val > 30)
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        prev_direction = np.roll(direction, 1).astype(float)
        prev_direction[0] = np.nan
        dir_change = direction != prev_direction
        adx_lt20 = adx_val < 20
        exit_mask = dir_change | adx_lt20
        signals[exit_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = signals == 1.0
        entry_short = signals == -1.0

        if entry_long.any():
            df.loc[entry_long, "bb_stop_long"] = close[entry_long] - (
                params["stop_atr_mult"] * atr[entry_long]
            )
            df.loc[entry_long, "bb_tp_long"] = close[entry_long] + (
                params["tp_atr_mult"] * atr[entry_long]
            )

        if entry_short.any():
            df.loc[entry_short, "bb_stop_short"] = close[entry_short] + (
                params["stop_atr_mult"] * atr[entry_short]
            )
            df.loc[entry_short, "bb_tp_short"] = close[entry_short] - (
                params["tp_atr_mult"] * atr[entry_short]
            )

        signals.iloc[:warmup] = 0.0
        return signals