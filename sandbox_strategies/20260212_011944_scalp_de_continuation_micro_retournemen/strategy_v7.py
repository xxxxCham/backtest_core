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
                default=70, min=50, max=90, step=1, description="RSI overbought threshold"
            ),
            "rsi_oversold": ParameterSpec(
                default=30, min=10, max=50, step=1, description="RSI oversold threshold"
            ),
            "stop_atr_mult": ParameterSpec(
                default=1.5, min=0.5, max=5.0, step=0.1, description="Stop‑loss ATR multiplier"
            ),
            "tp_atr_mult": ParameterSpec(
                default=3.0, min=1.0, max=10.0, step=0.1, description="Take‑profit ATR multiplier"
            ),
            "warmup": ParameterSpec(
                default=50, min=0, max=200, step=1, description="Warmup period (bars)"
            ),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        # Prepare price series
        close = np.nan_to_num(df["close"].values)

        # Bollinger bands
        bb = indicators["bollinger"]
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])
        lower = np.nan_to_num(bb["lower"])

        # RSI and its lagged version
        rsi = np.nan_to_num(indicators["rsi"])
        rsi_prev = np.concatenate((np.array([np.nan]), rsi[:-1]))

        # Parameters
        overbought = float(params.get("rsi_overbought", 70))
        oversold = float(params.get("rsi_oversold", 30))

        # Entry conditions
        long_entry = (close < lower) & (rsi_prev <= oversold) & (rsi > oversold) & (close > middle)
        short_entry = (close > upper) & (rsi_prev >= overbought) & (rsi < overbought) & (close < middle)

        # Exit conditions (generic, will be filtered by current position)
        exit_long = (close >= upper) | (rsi >= overbought) | (close <= lower)
        exit_short = (close <= lower) | (rsi <= oversold) | (close >= upper)

        # Signal generation with position tracking
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        position = 0  # 0 = flat, 1 = long, -1 = short

        for i in range(len(df)):
            if position == 0:
                if long_entry[i]:
                    position = 1
                elif short_entry[i]:
                    position = -1
            elif position == 1:
                if exit_long[i]:
                    position = 0
            elif position == -1:
                if exit_short[i]:
                    position = 0

            signals.iloc[i] = float(position)

        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        return signals