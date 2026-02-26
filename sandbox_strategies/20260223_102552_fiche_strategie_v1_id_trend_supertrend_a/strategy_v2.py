from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="trend_supertrend_with_ema")

    @property
    def required_indicators(self) -> List[str]:
        return ["supertrend", "adx", "ema", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "adx_period": 16,
            "atr_period": 14,
            "ema_period": 20,
            "leverage": 1,
            "stop_atr_mult": 1.25,
            "supertrend_atr_period": 18,
            "supertrend_multiplier": 2.5,
            "tp_atr_mult": 2.5,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "supertrend_atr_period": ParameterSpec(
                name="supertrend_atr_period",
                min_val=5,
                max_val=30,
                default=18,
                param_type="int",
                step=1,
            ),
            "supertrend_multiplier": ParameterSpec(
                name="supertrend_multiplier",
                min_val=1.0,
                max_val=5.0,
                default=2.5,
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
            "ema_period": ParameterSpec(
                name="ema_period",
                min_val=10,
                max_val=50,
                default=20,
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
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        warmup = int(params.get("warmup", 50))

        # Indicator arrays
        close = np.nan_to_num(df["close"].values, nan=0.0)
        ema = np.nan_to_num(indicators['ema'], nan=0.0)

        adx_val = np.nan_to_num(indicators['adx']["adx"], nan=0.0)
        direction = np.nan_to_num(indicators['supertrend']["direction"], nan=0.0)
        atr = np.nan_to_num(indicators['atr'], nan=0.0)

        # Long and short entry masks
        long_mask = (direction == 1.0) & (close > ema) & (adx_val > 25.0)
        short_mask = (direction == -1.0) & (close < ema) & (adx_val > 25.0)

        # Apply warmup
        signals.iloc[:warmup] = 0.0

        # Set entry signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ATR based SL/TP levels on entry bars
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params.get("stop_atr_mult", 1.25)
        tp_atr_mult = params.get("tp_atr_mult", 2.5)

        entry_long = signals == 1.0
        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_atr_mult * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_atr_mult * atr[entry_long]

        entry_short = signals == -1.0
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_atr_mult * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_atr_mult * atr[entry_short]

        # Exit conditions (used by the simulator)
        prev_dir = np.roll(direction, 1).astype(float)
        prev_dir[0] = np.nan
        dir_change = (direction != prev_dir) & (~np.isnan(prev_dir))

        adx_exit = adx_val < 20.0

        prev_close = np.roll(close, 1).astype(float)
        prev_close[0] = np.nan
        cross_any = (
            (close > ema) & (prev_close <= ema)
            | (close < ema) & (prev_close >= ema)
        )

        exit_mask = dir_change | adx_exit | cross_any
        # The simulator will handle exits; signals array remains unchanged for entries.

        signals.iloc[:warmup] = 0.0
        return signals