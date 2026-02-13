from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bolinger_supertrend_atr_breakout_with_rsi_filter")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "supertrend", "atr", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(param_type="int", min_value=50, max_value=90, step=5),
            "rsi_oversold": ParameterSpec(param_type="int", min_value=10, max_value=50, step=5),
            "rsi_period": ParameterSpec(param_type="int", min_value=5, max_value=30, step=5),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=5.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=3.0, max_value=10.0, step=0.5),
            "warmup": ParameterSpec(param_type="int", min_value=20, max_value=100, step=10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        bb = indicators["bollinger"]
        st = indicators["supertrend"]
        atr = indicators["atr"]
        rsi = indicators["rsi"]

        bb_upper = np.nan_to_num(bb["upper"])
        bb_lower = np.nan_to_num(bb["lower"])
        bb_middle = np.nan_to_num(bb["middle"])
        st_direction = np.nan_to_num(st["direction"])
        prev_bb_upper = np.roll(bb_upper, 1)
        prev_bb_lower = np.roll(bb_lower, 1)
        prev_close = np.roll(df["close"], 1)
        prev_st_direction = np.roll(st_direction, 1)
        prev_rsi = np.roll(rsi, 1)

        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 5.0)

        # Entry conditions
        entry_condition = (
            (df["close"] > bb_upper)
            & (prev_close <= prev_bb_upper)
            & (st_direction == 1)
            & (rsi < rsi_oversold)
            & (prev_rsi >= rsi_oversold)
        )

        # Exit conditions
        exit_condition = (
            (df["close"] < bb_lower)
            | (st_direction == -1)
        )

        # Generate signals
        entry_signals = pd.Series(0.0, index=df.index)
        entry_signals[entry_condition] = 1.0
        exit_signals = pd.Series(0.0, index=df.index)
        exit_signals[exit_condition] = -1.0

        # Combine signals
        signals = entry_signals + exit_signals

        return signals