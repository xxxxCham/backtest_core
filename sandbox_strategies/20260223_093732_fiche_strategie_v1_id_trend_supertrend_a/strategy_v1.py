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
            "stop_atr_mult": 1.25,
            "supertrend_atr_period": 5,
            "supertrend_multiplier": 2.0,
            "tp_atr_mult": 5.5,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "supertrend_atr_period": ParameterSpec(
                name="supertrend_atr_period",
                min_val=1,
                max_val=20,
                default=5,
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
                min_val=5,
                max_val=30,
                default=16,
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
                default=1.25,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=2.0,
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
        warmup = int(params.get("warmup", 50))

        # initialise signal series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # wrap indicator arrays
        st_dir = np.asarray(indicators['supertrend']["direction"], dtype=float)
        adx_val = np.asarray(indicators['adx']["adx"], dtype=float)
        atr = np.asarray(indicators['atr'], dtype=float)

        # entry conditions
        long_mask = (st_dir == 1) & (adx_val > 25)
        short_mask = (st_dir == -1) & (adx_val > 25)

        # exit condition: direction change or ADX below 20
        # compute direction change without NaNs
        dir_change = np.zeros(n, dtype=bool)
        dir_change[1:] = (st_dir[1:] != st_dir[:-1])
        exit_mask = dir_change | (adx_val < 20)

        # apply warmup
        signals.iloc[:warmup] = 0.0

        # set entry signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # set flat signals on exit
        signals[exit_mask] = 0.0

        # ATR‑based SL/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        close = df["close"].values
        stop_atr_mult = params.get("stop_atr_mult", 1.25)
        tp_atr_mult = params.get("tp_atr_mult", 5.5)

        entry_long_mask = signals == 1.0
        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - (
            stop_atr_mult * atr[entry_long_mask]
        )
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + (
            tp_atr_mult * atr[entry_long_mask]
        )

        entry_short_mask = signals == -1.0
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + (
            stop_atr_mult * atr[entry_short_mask]
        )
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - (
            tp_atr_mult * atr[entry_short_mask]
        )

        # ensure warmup period is flat
        signals.iloc[:warmup] = 0.0
        return signals