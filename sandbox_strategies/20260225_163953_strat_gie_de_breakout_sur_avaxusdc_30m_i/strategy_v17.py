from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_atr_adx_breakout")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "atr", "adx"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "adx_period": 14,
            "atr_period": 14,
            "leverage": 1,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 4.8,
            "trailing_atr_mult": 2.3,
            "warmup": 50,
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
            "adx_period": ParameterSpec(
                name="adx_period",
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
                max_val=10.0,
                default=4.8,
                param_type="float",
                step=0.1,
            ),
            "trailing_atr_mult": ParameterSpec(
                name="trailing_atr_mult",
                min_val=1.0,
                max_val=5.0,
                default=2.3,
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
        # initialize masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Extract indicator arrays
        close = df["close"].values
        bb = indicators['bollinger']
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])
        lower = np.nan_to_num(bb["lower"])
        atr = np.nan_to_num(indicators['atr'])
        adx = np.nan_to_num(indicators['adx']["adx"])

        # Cross helpers
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_upper = np.roll(upper, 1)
        prev_upper[0] = np.nan
        prev_lower = np.roll(lower, 1)
        prev_lower[0] = np.nan

        # Threshold for ADX
        adx_threshold = params.get("adx_threshold", 25)

        # Long entry: breakout above upper band with strong trend
        long_cross = (close > upper) & (prev_close <= prev_upper)
        long_mask = long_cross & (adx > adx_threshold)

        # Short entry: breakout below lower band with strong trend
        short_cross = (close < lower) & (prev_close >= prev_lower)
        short_mask = short_cross & (adx > adx_threshold)

        # Cooldown: prevent consecutive same‑direction entries
        cooldown = params.get("cooldown", 0)
        if cooldown > 0:
            long_shift = np.roll(long_mask, 1)
            long_shift[:cooldown] = False
            long_mask = long_mask & ~long_shift
            short_shift = np.roll(short_mask, 1)
            short_shift[:cooldown] = False
            short_mask = short_mask & ~short_shift

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ATR‑based SL/TP for entries
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - params["stop_atr_mult"] * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + params["tp_atr_mult"] * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + params["stop_atr_mult"] * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - params["tp_atr_mult"] * atr[short_mask]

        # Zero signals during warmup
        signals.iloc[:warmup] = 0.0
        return signals