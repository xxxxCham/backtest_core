from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_atr_scalp")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(
                name="rsi_overbought",
                type=int,
                default=70,
                min_value=60,
                max_value=90,
                description="RSI level considered overbought",
            ),
            "rsi_oversold": ParameterSpec(
                name="rsi_oversold",
                type=int,
                default=30,
                min_value=10,
                max_value=40,
                description="RSI level considered oversold",
            ),
            "rsi_period": ParameterSpec(
                name="rsi_period",
                type=int,
                default=14,
                min_value=5,
                max_value=30,
                description="Period for RSI calculation (informational)",
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                type=float,
                default=1.5,
                min_value=0.5,
                max_value=3.0,
                description="Stop‑loss distance as multiple of ATR",
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                type=float,
                default=3.0,
                min_value=1.0,
                max_value=5.0,
                description="Take‑profit distance as multiple of ATR",
            ),
            "warmup": ParameterSpec(
                name="warmup",
                type=int,
                default=50,
                min_value=0,
                max_value=200,
                description="Number of initial bars to set flat to avoid NaNs",
            ),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        # initialise flat signal series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # extract price series
        close = df["close"]

        # extract and clean indicators
        rsi_raw = indicators["rsi"]
        rsi = pd.Series(np.nan_to_num(rsi_raw), index=df.index)

        bb = indicators["bollinger"]
        upper = pd.Series(np.nan_to_num(bb["upper"]), index=df.index)
        lower = pd.Series(np.nan_to_num(bb["lower"]), index=df.index)

        atr_raw = indicators["atr"]
        atr = pd.Series(np.nan_to_num(atr_raw), index=df.index)

        # parameters
        rsi_ob = float(params.get("rsi_overbought", 70))
        rsi_os = float(params.get("rsi_oversold", 30))
        stop_mult = float(params.get("stop_atr_mult", 1.5))
        tp_mult = float(params.get("tp_atr_mult", 3.0))
        warmup = int(params.get("warmup", 50))

        # entry conditions
        long_entry = (close <= lower) & (rsi < rsi_os)
        short_entry = (close >= upper) & (rsi > rsi_ob)

        signals[long_entry] = 1.0
        signals[short_entry] = -1.0

        # warmup protection
        if warmup > 0:
            signals.iloc[:warmup] = 0.0

        # build position series (forward filled)
        position = signals.replace(0, np.nan).ffill().fillna(0.0)

        # entry price series (price at the moment of entry, then forward filled)
        entry_price = pd.Series(np.where(signals != 0, close, np.nan), index=df.index)
        entry_price = entry_price.replace(0, np.nan).ffill()

        # stop‑loss and take‑profit levels based on entry price and ATR
        stop_price_long = entry_price - stop_mult * atr
        tp_price_long = entry_price + tp_mult * atr

        stop_price_short = entry_price + stop_mult * atr
        tp_price_short = entry_price - tp_mult * atr

        # exit conditions
        long_exit = (position == 1.0) & (
            (close >= upper) |
            (close <= stop_price_long) |
            (close >= tp_price_long)
        )
        short_exit = (position == -1.0) & (
            (close <= lower) |
            (close >= stop_price_short) |
            (close <= tp_price_short)
        )

        # apply exits: set signal to flat where exit condition met
        signals[long_exit] = 0.0
        signals[short_exit] = 0.0

        return signals