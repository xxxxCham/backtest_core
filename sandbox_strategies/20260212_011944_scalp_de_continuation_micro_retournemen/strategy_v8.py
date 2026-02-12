from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_atr_continuation")

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
                type=float,
                default=70,
                min=50,
                max=90,
                description="RSI level considered overbought",
            ),
            "rsi_oversold": ParameterSpec(
                name="rsi_oversold",
                type=float,
                default=30,
                min=10,
                max=50,
                description="RSI level considered oversold",
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                type=float,
                default=1.5,
                min=0.5,
                max=5.0,
                description="ATR multiplier for stop‑loss",
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                type=float,
                default=3.0,
                min=1.0,
                max=10.0,
                description="ATR multiplier for take‑profit",
            ),
            "warmup": ParameterSpec(
                name="warmup",
                type=int,
                default=50,
                min=0,
                max=200,
                description="Number of initial bars to set flat",
            ),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        # Prepare price series
        close = np.nan_to_num(df["close"].values.astype(float))

        # Load and sanitize indicators
        rsi = np.nan_to_num(indicators["rsi"].astype(float))
        bb = indicators["bollinger"]
        upper = np.nan_to_num(bb["upper"].astype(float))
        lower = np.nan_to_num(bb["lower"].astype(float))
        atr = np.nan_to_num(indicators["atr"].astype(float))

        # Previous RSI (shifted by one bar)
        rsi_prev = np.concatenate(([np.nan], rsi[:-1]))
        rsi_prev = np.nan_to_num(rsi_prev)

        # Parameters
        overbought = float(params.get("rsi_overbought", 70))
        oversold = float(params.get("rsi_oversold", 30))

        # Entry conditions
        long_entry = (close <= lower) & (rsi_prev <= oversold) & (rsi > oversold)
        short_entry = (close >= upper) & (rsi_prev >= overbought) & (rsi < overbought)

        # Exit conditions
        long_exit = (close >= upper) | (rsi >= overbought)
        short_exit = (close <= lower) | (rsi <= oversold)

        # Build position series iteratively
        signals_arr = np.zeros_like(close, dtype=np.float64)
        position = 0.0

        for i in range(len(close)):
            if position == 0.0:
                if long_entry[i]:
                    position = 1.0
                elif short_entry[i]:
                    position = -1.0
            else:
                if position == 1.0 and long_exit[i]:
                    position = 0.0
                elif position == -1.0 and short_exit[i]:
                    position = 0.0
            signals_arr[i] = position

        # Convert to pandas Series with original index
        signals = pd.Series(signals_arr, index=df.index, dtype=np.float64)

        # Warmup protection
        warmup = int(params.get("warmup", 50))
        if warmup > 0:
            signals.iloc[:warmup] = 0.0

        return signals