from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_atr_sc_aligned")

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
            "rsi_overbought": ParameterSpec(default=70, min=50, max=90, description="RSI overbought threshold"),
            "rsi_oversold": ParameterSpec(default=30, min=10, max=50, description="RSI oversold threshold"),
            "rsi_period": ParameterSpec(default=14, min=5, max=30, description="RSI look‑back period"),
            "stop_atr_mult": ParameterSpec(default=1.5, min=0.5, max=5.0, description="ATR multiplier for stop loss"),
            "tp_atr_mult": ParameterSpec(default=3.0, min=1.0, max=10.0, description="ATR multiplier for take profit"),
            "warmup": ParameterSpec(default=50, min=0, max=200, description="Warmup bars before signals"),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        close = df["close"].values

        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        lower = np.nan_to_num(bb["lower"])
        middle = np.nan_to_num(bb["middle"])
        upper = np.nan_to_num(bb["upper"])
        atr = np.nan_to_num(indicators["atr"])

        # previous RSI (shifted by one bar)
        rsi_prev = np.empty_like(rsi)
        rsi_prev[0] = np.nan
        rsi_prev[1:] = rsi[:-1]

        oversold = params.get("rsi_oversold", 30)
        overbought = params.get("rsi_overbought", 70)
        stop_mult = params.get("stop_atr_mult", 1.5)
        tp_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))

        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        position = 0.0
        entry_price = np.nan
        stop_price = np.nan
        tp_price = np.nan

        for i in range(len(close)):
            if i < warmup:
                signals.iloc[i] = 0.0
                continue

            price = close[i]

            if position == 0.0:
                long_cond = (
                    price > middle[i]
                    and price <= lower[i]
                    and rsi[i] > oversold
                    and rsi_prev[i] <= oversold
                )
                short_cond = (
                    price < middle[i]
                    and price >= upper[i]
                    and rsi[i] < overbought
                    and rsi_prev[i] >= overbought
                )

                if long_cond:
                    position = 1.0
                    entry_price = price
                    stop_price = entry_price - stop_mult * atr[i]
                    tp_price = entry_price + tp_mult * atr[i]
                    signals.iloc[i] = position
                    continue
                elif short_cond:
                    position = -1.0
                    entry_price = price
                    stop_price = entry_price + stop_mult * atr[i]
                    tp_price = entry_price - tp_mult * atr[i]
                    signals.iloc[i] = position
                    continue
                else:
                    signals.iloc[i] = 0.0
                    continue

            elif position == 1.0:
                exit_long = (
                    price >= upper[i] or price <= stop_price or price >= tp_price
                )
                if exit_long:
                    position = 0.0
                    entry_price = np.nan
                    stop_price = np.nan
                    tp_price = np.nan
                    signals.iloc[i] = 0.0
                else:
                    signals.iloc[i] = 1.0

            else:  # position == -1.0
                exit_short = (
                    price <= lower[i] or price >= stop_price or price <= tp_price
                )
                if exit_short:
                    position = 0.0
                    entry_price = np.nan
                    stop_price = np.nan
                    tp_price = np.nan
                    signals.iloc[i] = 0.0
                else:
                    signals.iloc[i] = -1.0

        return signals