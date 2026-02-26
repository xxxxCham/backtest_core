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
        return ["supertrend", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "atr_period": 14,
            "leverage": 1,
            "rsi_overbought": 60,
            "rsi_oversold": 40,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "supertrend_multiplier": 2.5,
            "supertrend_period": 15,
            "tp_atr_mult": 3.0,
            "warmup": 30,
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
            "rsi_overbought": ParameterSpec(
                name="rsi_overbought",
                min_val=50,
                max_val=80,
                default=60,
                param_type="int",
                step=1,
            ),
            "rsi_oversold": ParameterSpec(
                name="rsi_oversold",
                min_val=20,
                max_val=50,
                default=40,
                param_type="int",
                step=1,
            ),
            "supertrend_period": ParameterSpec(
                name="supertrend_period",
                min_val=5,
                max_val=30,
                default=15,
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
                default=1.5,
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
            "warmup": ParameterSpec(
                name="warmup",
                min_val=10,
                max_val=50,
                default=30,
                param_type="int",
                step=1,
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

        # Initialize signals
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Retrieve indicator arrays
        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])
        st = indicators['supertrend']
        direction = np.nan_to_num(st["direction"])

        # Parameters
        rsi_overbought = float(params.get("rsi_overbought", 60))
        rsi_oversold = float(params.get("rsi_oversold", 40))
        stop_atr_mult = float(params.get("stop_atr_mult", 1.5))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.0))

        # Entry conditions
        long_mask = (direction == 1) & (rsi > rsi_overbought)
        short_mask = (direction == -1) & (rsi < rsi_oversold)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions: direction change or RSI crossing 50
        prev_dir = np.roll(direction, 1).astype(float)
        prev_dir[0] = np.nan
        dir_change = direction != prev_dir

        rsi_50 = np.full(n, 50.0)
        prev_rsi = np.roll(rsi, 1).astype(float)
        prev_rsi[0] = np.nan
        cross_up = (rsi > rsi_50) & (prev_rsi <= rsi_50)
        cross_down = (rsi < rsi_50) & (prev_rsi >= rsi_50)
        rsi_cross = cross_up | cross_down

        exit_mask = dir_change | rsi_cross
        signals[exit_mask] = 0.0

        # ATR-based SL/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        close = df["close"].values
        long_entry = signals == 1.0
        short_entry = signals == -1.0

        df.loc[long_entry, "bb_stop_long"] = close[long_entry] - stop_atr_mult * atr[long_entry]
        df.loc[long_entry, "bb_tp_long"] = close[long_entry] + tp_atr_mult * atr[long_entry]
        df.loc[short_entry, "bb_stop_short"] = close[short_entry] + stop_atr_mult * atr[short_entry]
        df.loc[short_entry, "bb_tp_short"] = close[short_entry] - tp_atr_mult * atr[short_entry]

        signals.iloc[:warmup] = 0.0
        return signals