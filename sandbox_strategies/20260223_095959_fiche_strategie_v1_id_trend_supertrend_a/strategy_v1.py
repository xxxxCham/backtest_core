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
            "adx_period": 16,
            "atr_period": 14,
            "leverage": 1,
            "stop_atr_mult": 1.75,
            "supertrend_atr_period": 18,
            "supertrend_multiplier": 2.0,
            "tp_atr_mult": 4.5,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "supertrend_atr_period": ParameterSpec(
                name="supertrend_atr_period",
                min_val=10,
                max_val=30,
                default=18,
                param_type="int",
                step=1,
            ),
            "supertrend_multiplier": ParameterSpec(
                name="supertrend_multiplier",
                min_val=1.0,
                max_val=3.0,
                default=2.0,
                param_type="float",
                step=0.1,
            ),
            "adx_period": ParameterSpec(
                name="adx_period",
                min_val=10,
                max_val=30,
                default=16,
                param_type="int",
                step=1,
            ),
            "atr_period": ParameterSpec(
                name="atr_period",
                min_val=10,
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
                default=4.5,
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

        # Boolean masks for entry and exit
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Extract indicator arrays
        # Ensure direction is float to allow NaN handling
        direction = np.asarray(indicators['supertrend']["direction"], dtype=float)
        adx_val = np.nan_to_num(np.asarray(indicators['adx']["adx"], dtype=float))
        atr = np.nan_to_num(np.asarray(indicators['atr'], dtype=float))

        # Entry conditions
        long_mask = (direction == 1) & (adx_val > 30)
        short_mask = (direction == -1) & (adx_val > 30)

        # Exit condition: direction change or weak ADX
        prev_direction = np.roll(direction, 1).astype(float)
        prev_direction[0] = np.nan
        direction_change = (direction != prev_direction) & (~np.isnan(prev_direction))
        exit_mask = direction_change | (adx_val < 25)

        # Initialize signals series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Apply warmup period
        signals.iloc[:warmup] = 0.0

        # Set entry signals
        idx = np.arange(n) >= warmup
        entry_long = long_mask & idx
        entry_short = short_mask & idx
        signals[entry_long] = 1.0
        signals[entry_short] = -1.0

        # Set exit signals
        exit_bar = exit_mask & idx
        signals[exit_bar] = 0.0

        # Prepare SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Compute SL/TP levels on entry bars
        close = df["close"].values
        stop_atr_mult = float(params.get("stop_atr_mult", 1.75))
        tp_atr_mult = float(params.get("tp_atr_mult", 4.5))

        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_atr_mult * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_atr_mult * atr[entry_long]

        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_atr_mult * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_atr_mult * atr[entry_short]

        return signals