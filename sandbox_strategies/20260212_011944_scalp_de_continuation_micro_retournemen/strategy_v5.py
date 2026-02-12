from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_atr_sc_continuation")

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
                default=70, min=50, max=90, description="RSI overbought threshold"
            ),
            "rsi_oversold": ParameterSpec(
                default=30, min=10, max=50, description="RSI oversold threshold"
            ),
            "rsi_period": ParameterSpec(
                default=14, min=5, max=30, description="RSI look‑back period (for reference only)"
            ),
            "stop_atr_mult": ParameterSpec(
                default=1.5, min=0.5, max=5.0, description="ATR multiplier for stop‑loss"
            ),
            "tp_atr_mult": ParameterSpec(
                default=3.0, min=1.0, max=10.0, description="ATR multiplier for take‑profit"
            ),
            "warmup": ParameterSpec(
                default=50, min=0, max=200, description="Number of initial bars set to flat"
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

        # price series
        price = np.nan_to_num(df["close"].values)

        # Bollinger bands
        bb = indicators["bollinger"]
        lower = np.nan_to_num(bb["lower"])
        middle = np.nan_to_num(bb["middle"])
        upper = np.nan_to_num(bb["upper"])

        # RSI and its previous value
        rsi = np.nan_to_num(indicators["rsi"])
        rsi_prev = np.concatenate(([np.nan], rsi[:-1]))
        rsi_prev = np.nan_to_num(rsi_prev)

        # parameters
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)

        # LONG entry: pull‑back toward middle band from below, RSI crossing up from oversold
        long_cond = (
            (price < middle)
            & (price >= lower)
            & (rsi_prev <= rsi_oversold)
            & (rsi > rsi_oversold)
        )

        # SHORT entry: pull‑back toward middle band from above, RSI crossing down from overbought
        short_cond = (
            (price > middle)
            & (price <= upper)
            & (rsi_prev >= rsi_overbought)
            & (rsi < rsi_overbought)
        )

        signals[long_cond] = 1.0
        signals[short_cond] = -1.0

        # warmup protection
        warmup = int(params.get("warmup", 50))
        if warmup > 0:
            signals.iloc[:warmup] = 0.0

        return signals