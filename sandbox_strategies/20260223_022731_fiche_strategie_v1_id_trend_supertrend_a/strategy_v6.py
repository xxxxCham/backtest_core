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
            "leverage": 1,
            "rsi_overbought": 65,
            "rsi_oversold": 35,
            "rsi_period": 14,
            "stop_atr_mult": 1.25,
            "tp_atr_mult": 5.5,
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
            "rsi_oversold": ParameterSpec(
                name="rsi_oversold",
                min_val=20,
                max_val=50,
                default=35,
                param_type="int",
                step=1,
            ),
            "rsi_overbought": ParameterSpec(
                name="rsi_overbought",
                min_val=50,
                max_val=80,
                default=65,
                param_type="int",
                step=1,
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                min_val=0.5,
                max_val=3.0,
                default=1.25,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=2.0,
                max_val=6.0,
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

        # Indicator arrays – cast to float to allow NaNs
        direction = np.asarray(indicators['supertrend']["direction"], dtype=float)
        rsi = np.asarray(indicators['rsi'], dtype=float)
        atr = np.asarray(indicators['atr'], dtype=float)

        # Entry conditions
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        long_mask = (direction == 1) & (rsi < rsi_oversold)
        short_mask = (direction == -1) & (rsi > rsi_overbought)

        # Exit conditions: direction change or RSI crossing 50
        prev_direction = np.roll(direction, 1)
        prev_direction[0] = np.nan
        direction_change = (direction != prev_direction) & (~np.isnan(prev_direction))

        prev_rsi = np.roll(rsi, 1)
        prev_rsi[0] = np.nan
        cross_up_50 = (rsi > 50) & (prev_rsi <= 50)
        cross_down_50 = (rsi < 50) & (prev_rsi >= 50)
        rsi_cross_50 = cross_up_50 | cross_down_50

        exit_mask = direction_change | rsi_cross_50

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # Warm‑up period
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # ATR‑based stop‑loss and take‑profit columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        close = df["close"].values
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        return signals