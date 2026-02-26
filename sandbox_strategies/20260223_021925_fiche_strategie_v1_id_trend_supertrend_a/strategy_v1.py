from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="trend_supertrend_optimized")

    @property
    def required_indicators(self) -> List[str]:
        return ["supertrend", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"leverage": 1, "stop_atr_mult": 1.75, "tp_atr_mult": 2.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "supertrend_atr_period": ParameterSpec(
                name="supertrend_atr_period",
                min_val=5,
                max_val=30,
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
                max_val=50,
                default=23,
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
                default=1.75,
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
        n = len(df)
        warmup = int(params.get("warmup", 50))
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Wrap indicator arrays as floats to safely handle NaNs
        st_dir = np.asarray(indicators['supertrend']["direction"], dtype=float)
        adx_val = np.asarray(indicators['adx']["adx"], dtype=float)
        atr_val = np.asarray(indicators['atr'], dtype=float)
        close = df["close"].values

        # Entry conditions
        long_mask = (st_dir == 1.0) & (adx_val > 25)
        short_mask = (st_dir == -1.0) & (adx_val > 25)

        # Exit conditions: trend change or weak ADX
        prev_dir = np.roll(st_dir, 1).astype(float)
        prev_dir[0] = np.nan
        dir_change = (st_dir != prev_dir)
        exit_mask = dir_change | (adx_val < 15)

        # Apply warmup
        signals.iloc[:warmup] = 0.0

        # Set entry signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Set exit signals (flatten)
        signals[exit_mask] = 0.0

        # ATR-based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_mult = params.get("stop_atr_mult", 1.75)
        tp_mult = params.get("tp_atr_mult", 2.5)

        # Compute levels only on entry bars
        entry_long = signals == 1.0
        entry_short = signals == -1.0
        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_mult * atr_val[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_mult * atr_val[entry_long]
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_mult * atr_val[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_mult * atr_val[entry_short]

        # Re-apply warmup to ensure no signals during warmup period
        signals.iloc[:warmup] = 0.0
        return signals