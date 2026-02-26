from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="trend_supertrend_rsi")

    @property
    def required_indicators(self) -> List[str]:
        return ["supertrend", "adx", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "adx_period": 23,
            "leverage": 1,
            "rsi_period": 14,
            "stop_atr_mult": 1.75,
            "supertrend_atr_period": 16,
            "supertrend_multiplier": 2.0,
            "tp_atr_mult": 2.5,
            "warmup": 30,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(
                name="rsi_period", min_val=5, max_val=30, default=14, param_type="int", step=1
            ),
            "adx_period": ParameterSpec(
                name="adx_period", min_val=10, max_val=50, default=23, param_type="int", step=1
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult", min_val=0.5, max_val=4.0, default=1.75, param_type="float", step=0.1
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult", min_val=1.0, max_val=5.0, default=2.5, param_type="float", step=0.1
            ),
            "leverage": ParameterSpec(
                name="leverage", min_val=1, max_val=2, default=1, param_type="int", step=1
            ),
        }

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)

        warmup = int(params.get("warmup", 50))

        # Pull indicator arrays
        st = np.nan_to_num(indicators['supertrend']["direction"])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        rsi_val = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])

        # Long/short entry logic
        long_mask = (st == 1) & (adx_val > 25) & (rsi_val > 50)
        short_mask = (st == -1) & (adx_val > 25) & (rsi_val < 50)

        # Trend direction change detection
        prev_st = np.roll(st, 1)
        dir_change = (st != prev_st) & (np.arange(n) > 0)

        # RSI cross 50 detection
        rsi_cross_up = (rsi_val > 50) & (np.roll(rsi_val, 1) <= 50)
        rsi_cross_down = (rsi_val < 50) & (np.roll(rsi_val, 1) >= 50)
        rsi_cross = rsi_cross_up | rsi_cross_down

        # Exit conditions
        adx_low = adx_val < 20
        exit_mask = dir_change | adx_low | rsi_cross

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # Warm‑up period
        signals.iloc[:warmup] = 0.0

        # Prepare columns for stops and targets
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_mult = params.get("stop_atr_mult", 1.75)
        tp_mult = params.get("tp_atr_mult", 2.5)

        close = df["close"].values

        long_entry = signals == 1.0
        short_entry = signals == -1.0

        df.loc[long_entry, "bb_stop_long"] = close[long_entry] - stop_mult * atr[long_entry]
        df.loc[long_entry, "bb_tp_long"] = close[long_entry] + tp_mult * atr[long_entry]

        df.loc[short_entry, "bb_stop_short"] = close[short_entry] + stop_mult * atr[short_entry]
        df.loc[short_entry, "bb_tp_short"] = close[short_entry] - tp_mult * atr[short_entry]

        # Ensure warm‑up signals remain zero
        signals.iloc[:warmup] = 0.0

        return signals