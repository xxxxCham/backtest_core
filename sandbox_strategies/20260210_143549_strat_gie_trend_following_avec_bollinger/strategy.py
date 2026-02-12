from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    """
    Auto-generated strategy: BollingerRSI Trend Strategy
    Objective: Stratégie trend-following avec Bollinger + RSI
    Indicators: bollinger, rsi
    """

    def __init__(self):
        super().__init__(name="BollingerRSI Trend Strategy")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"bollinger_period": 20, "rsi_period": 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "bollinger_period": ParameterSpec(
                type_=int,
                default=20,
                bounds=(5, 100),
                description="Period for Bollinger Bands calculation."
            ),
            "rsi_period": ParameterSpec(
                type_=int,
                default=14,
                bounds=(2, 100),
                description="Period for RSI calculation."
            )
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Extract Bollinger Bands
        boll_upper, boll_mid, boll_lower = indicators["bollinger"]
        rsi = np.nan_to_num(indicators["rsi"])

        for i in range(n):
            if i == 0:
                signals[i] = 0.0
                continue

            current_close = df.iloc[i]["close"]

            # Entry conditions
            long_entry = (
                current_close > boll_upper[i]
                and rsi[i] < 30
            )
            short_entry = (
                current_close < boll_lower[i]
                and rsi[i] > 70
            )

            # Exit conditions
            # Check if price touches opposite band
            is_long = signals[i-1] == 1.0
            is_short = signals[i-1] == -1.0

            exit_price_long = current_close < boll_lower[i]
            exit_price_short = current_close > boll_upper[i]

            # Check RSI divergence for exit
            prev_rsi_high = rsi[i-1]
            if (rsi[i] > prev_rsi_high) and is_long:
                signals[i] = -1.0
                continue

            if (rsi[i] < prev_rsi_high) and is_short:
                signals[i] = 1.0
                continue

            # Default to previous signal unless new entry condition met
            if long_entry:
                signals[i] = 1.0
            elif short_entry:
                signals[i] = -1.0
            else:
                signals[i] = signals[i-1]

        return signals