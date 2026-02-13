from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="macd_rsi_atr_scalper_v2")

    @property
    def required_indicators(self) -> List[str]:
        return ["macd", "rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(param_type="int", min_value=60, max_value=80, step=5),
            "rsi_oversold": ParameterSpec(param_type="int", min_value=10, max_value=30, step=5),
            "rsi_period": ParameterSpec(param_type="int", min_value=10, max_value=20, step=5),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=2.0, step=0.25),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=4.0, step=0.5),
            "warmup": ParameterSpec(param_type="int", min_value=30, max_value=100, step=10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Extract indicators
        macd = indicators["macd"]
        rsi = indicators["rsi"]
        bb = indicators["bollinger"]
        atr = indicators["atr"]

        # Prepare arrays
        macd_histogram = np.nan_to_num(macd["histogram"])
        macd_histogram_shifted = np.roll(macd_histogram, 1)
        macd_histogram_shifted[0] = 0.0

        rsi = np.nan_to_num(rsi)
        bb_upper = np.nan_to_num(bb["upper"])
        bb_middle = np.nan_to_num(bb["middle"])
        bb_lower = np.nan_to_num(bb["lower"])

        # RSI trend
        rsi_trend = np.zeros_like(rsi)
        rsi_trend[1:] = np.sign(rsi[1:] - rsi[:-1])

        # Entry conditions
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]

        # Long entry conditions
        long_condition = (
            (macd_histogram > 0) &
            (macd_histogram_shifted < 0) &
            (rsi < rsi_oversold) &
            (df["close"] > bb_middle) &
            (rsi_trend > 0)
        )

        # Short entry conditions
        short_condition = (
            (macd_histogram < 0) &
            (macd_histogram_shifted > 0) &
            (rsi > rsi_overbought) &
            (df["close"] < bb_middle) &
            (rsi_trend < 0)
        )

        # Exit conditions (MACD crossover)
        exit_condition = (
            (macd_histogram > 0) &
            (macd_histogram_shifted < 0) |
            (macd_histogram < 0) &
            (macd_histogram_shifted > 0)
        )

        # Apply signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0

        # Exit on crossover
        signals[exit_condition] = 0.0

        return signals