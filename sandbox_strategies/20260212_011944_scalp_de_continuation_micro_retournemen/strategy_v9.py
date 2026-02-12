from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_trend_pullback")

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
                default=70,
                min=50,
                max=90,
                type=int,
                description="RSI overbought threshold",
            ),
            "rsi_oversold": ParameterSpec(
                default=30,
                min=10,
                max=50,
                type=int,
                description="RSI oversold threshold",
            ),
            "rsi_period": ParameterSpec(
                default=14,
                min=5,
                max=30,
                type=int,
                description="RSI calculation period (pre‑computed)",
            ),
            "stop_atr_mult": ParameterSpec(
                default=1.5,
                min=0.5,
                max=5.0,
                type=float,
                description="Stop‑loss multiplier of ATR",
            ),
            "tp_atr_mult": ParameterSpec(
                default=3.0,
                min=1.0,
                max=10.0,
                type=float,
                description="Take‑profit multiplier of ATR",
            ),
            "warmup": ParameterSpec(
                default=50,
                min=0,
                max=200,
                type=int,
                description="Number of initial bars set to flat",
            ),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        # Prepare price series
        close = df["close"].values.astype(np.float64)

        # Load and clean indicators
        rsi_raw = indicators["rsi"]
        rsi = np.nan_to_num(rsi_raw)

        bb = indicators["bollinger"]
        lower = np.nan_to_num(bb["lower"])
        middle = np.nan_to_num(bb["middle"])
        upper = np.nan_to_num(bb["upper"])

        # ATR (not used directly in signal generation but kept for completeness)
        _ = np.nan_to_num(indicators["atr"])

        # Shifted RSI for previous‑bar comparison
        rsi_prev = np.concatenate(([np.nan], rsi[:-1]))
        rsi_prev = np.nan_to_num(rsi_prev)

        # Parameters
        overbought = float(params.get("rsi_overbought", 70))
        oversold = float(params.get("rsi_oversold", 30))
        warmup = int(params.get("warmup", 50))

        # Initialise signals array
        signals_arr = np.zeros_like(close, dtype=np.float64)
        position = 0.0

        for i in range(len(close)):
            if i < warmup:
                signals_arr[i] = 0.0
                continue

            if position == 0.0:
                long_cond = (
                    (close[i] < lower[i])
                    & (rsi_prev[i] <= oversold)
                    & (rsi[i] > oversold)
                    & (close[i] > middle[i])
                )
                short_cond = (
                    (close[i] > upper[i])
                    & (rsi_prev[i] >= overbought)
                    & (rsi[i] < overbought)
                    & (close[i] < middle[i])
                )
                if long_cond:
                    position = 1.0
                elif short_cond:
                    position = -1.0
            else:
                if position == 1.0:
                    exit_long = (close[i] >= upper[i]) | (rsi[i] >= overbought)
                    if exit_long:
                        position = 0.0
                elif position == -1.0:
                    exit_short = (close[i] <= lower[i]) | (rsi[i] <= oversold)
                    if exit_short:
                        position = 0.0

            signals_arr[i] = position

        return pd.Series(signals_arr, index=df.index, dtype=np.float64)