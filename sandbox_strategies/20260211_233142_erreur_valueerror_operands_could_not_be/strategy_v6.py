from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="trend_rsi_bollinger")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(name="RSI Overbought", type=float, min=20, max=90),
            "rsi_oversold": ParameterSpec(name="RSI Oversold", type=float, min=10, max=80),
            "rsi_period": ParameterSpec(name="RSI Period", type=int, min=5, max=50)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        rsi = np.nan_to_num(indicators["rsi"])
        bollinger = indicators["bollinger"]
        upper_bollinger = np.nan_to_num(bollinger["upper"])
        lower_bollinger = np.nan_to_num(bollinger["lower"])

        close_prices = df['close'].to_numpy()

        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)

        long_entry = (close_prices > upper_bollinger) & (rsi < rsi_oversold)
        short_entry = (close_prices < lower_bollinger) & (rsi > rsi_overbought)

        signals[long_entry] = 1.0
        signals[short_entry] = -1.0

        return signals