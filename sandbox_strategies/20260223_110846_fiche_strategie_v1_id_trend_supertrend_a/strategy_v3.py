from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="trend_supertrend_rsi_filter")

    @property
    def required_indicators(self) -> List[str]:
        return ["supertrend", "adx", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "leverage": 1,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(
                name="rsi_period",
                min_val=5,
                max_val=50,
                default=14,
                param_type="int",
                step=1,
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                min_val=0.5,
                max_val=4.0,
                default=1.5,
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
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=2.0,
                max_val=4.5,
                default=3.0,
                param_type="float",
                step=0.1,
            ),
        }

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        n = len(df)
        warmup = int(params.get("warmup", 50))

        # Initialize signal series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Ensure indicator arrays are float to allow NaN assignment
        supertrend_dir = np.array(indicators['supertrend']["direction"], dtype=float)
        adx_val = np.array(indicators['adx']["adx"], dtype=float)
        rsi_val = np.array(indicators['rsi'], dtype=float)
        atr_val = np.array(indicators['atr'], dtype=float)
        close = df["close"].values

        # Long / short entry conditions
        long_mask = (supertrend_dir == 1) & (adx_val > 30) & (rsi_val < 70)
        short_mask = (supertrend_dir == -1) & (adx_val > 30) & (rsi_val > 30)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        prev_dir = np.roll(supertrend_dir, 1)
        prev_dir[0] = np.nan  # safe because supertrend_dir is float
        dir_change = supertrend_dir != prev_dir
        exit_mask = dir_change | (adx_val < 25)
        signals[exit_mask] = 0.0

        # Warm‑up protection
        signals.iloc[:warmup] = 0.0

        # ATR‑based stop/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        long_entry_mask = signals == 1.0
        short_entry_mask = signals == -1.0

        df.loc[long_entry_mask, "bb_stop_long"] = (
            close[long_entry_mask] - params["stop_atr_mult"] * atr_val[long_entry_mask]
        )
        df.loc[long_entry_mask, "bb_tp_long"] = (
            close[long_entry_mask] + params["tp_atr_mult"] * atr_val[long_entry_mask]
        )
        df.loc[short_entry_mask, "bb_stop_short"] = (
            close[short_entry_mask] + params["stop_atr_mult"] * atr_val[short_entry_mask]
        )
        df.loc[short_entry_mask, "bb_tp_short"] = (
            close[short_entry_mask] - params["tp_atr_mult"] * atr_val[short_entry_mask]
        )

        return signals