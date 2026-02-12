from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_pullback_atr")

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
            "stop_atr_mult": ParameterSpec(
                default=1.5, min=0.5, max=5.0, description="Stop‑loss ATR multiplier"
            ),
            "tp_atr_mult": ParameterSpec(
                default=3.0, min=1.0, max=10.0, description="Take‑profit ATR multiplier"
            ),
            "warmup": ParameterSpec(
                default=50, min=0, max=200, description="Warm‑up period (bars)"
            ),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        # price series
        close = np.nan_to_num(df["close"].values.astype(float))

        # indicator arrays
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])
        bb = indicators["bollinger"]
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])
        lower = np.nan_to_num(bb["lower"])

        # previous RSI (shifted by one bar)
        rsi_prev = np.roll(rsi, 1)
        rsi_prev[0] = np.nan
        rsi_prev = np.nan_to_num(rsi_prev)

        # parameters
        rsi_ob = float(params.get("rsi_overbought", 70))
        rsi_os = float(params.get("rsi_oversold", 30))
        stop_mult = float(params.get("stop_atr_mult", 1.5))
        tp_mult = float(params.get("tp_atr_mult", 3.0))
        warmup = int(params.get("warmup", 50))

        # initialise signals
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        position = 0.0
        entry_price = 0.0

        for i in range(len(df)):
            # warm‑up protection
            if i < warmup:
                signals.iloc[i] = 0.0
                continue

            # entry conditions when flat
            if position == 0.0:
                long_cond = (close[i] < middle[i]) & (rsi_prev[i] <= rsi_os) & (rsi[i] > rsi_os)
                short_cond = (close[i] > middle[i]) & (rsi_prev[i] >= rsi_ob) & (rsi[i] < rsi_ob)

                if long_cond:
                    position = 1.0
                    entry_price = close[i]
                elif short_cond:
                    position = -1.0
                    entry_price = close[i]

            else:
                # calculate dynamic stop‑loss and take‑profit levels
                if position == 1.0:
                    stop_price = entry_price - stop_mult * atr[i]
                    tp_price = entry_price + tp_mult * atr[i]
                    exit_cond = (
                        (close[i] >= upper[i])
                        | (rsi[i] >= rsi_ob)
                        | (close[i] <= stop_price)
                        | (close[i] >= tp_price)
                    )
                    if exit_cond:
                        position = 0.0
                else:  # short position
                    stop_price = entry_price + stop_mult * atr[i]
                    tp_price = entry_price - tp_mult * atr[i]
                    exit_cond = (
                        (close[i] <= lower[i])
                        | (rsi[i] <= rsi_os)
                        | (close[i] >= stop_price)
                        | (close[i] <= tp_price)
                    )
                    if exit_cond:
                        position = 0.0

            signals.iloc[i] = position

        # enforce warm‑up flat period
        signals.iloc[:warmup] = 0.0
        return signals