from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    """
    Auto-generated strategy: BollingerRSI_Trend_Following
    Objective: Stratégie trend-following avec Bollinger + RSI
    Indicators: bollinger, rsi
    """

    def __init__(self):
        super().__init__(name="BollingerRSI_Trend_Following")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "bollinger_period": 20,
            "bollinger_std_dev": 2,
            "rsi_period": 14,
            "exit_rsi_threshold": 30,
            "sl_percent": 5,
            "tp_percent": 10
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "bollinger_period": ParameterSpec(
                type=int,
                min=2,
                max=50,
                default=20,
                description="Period for Bollinger Bands"
            ),
            "bollinger_std_dev": ParameterSpec(
                type=int,
                min=1,
                max=4,
                default=2,
                description="Standard deviations for Bollinger Bands"
            ),
            "rsi_period": ParameterSpec(
                type=int,
                min=2,
                max=50,
                default=14,
                description="Period for RSI indicator"
            ),
            "exit_rsi_threshold": ParameterSpec(
                type=int,
                min=20,
                max=80,
                default=30,
                description="RSI threshold for exit signal"
            ),
            "sl_percent": ParameterSpec(
                type=float,
                min=1.0,
                max=20.0,
                default=5.0,
                description="Stop-loss percentage"
            ),
            "tp_percent": ParameterSpec(
                type=float,
                min=1.0,
                max=20.0,
                default=10.0,
                description="Take-profit percentage"
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

        bollinger = indicators["bollinger"]
        upper_b = bollinger[0]
        middle_b = bollinger[1]
        lower_b = bollinger[2]

        rsi = indicators["rsi"]

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # Entry conditions for LONG
        entry_long = (
            (close > upper_b) &
            (rsi < params["exit_rsi_threshold"])
        )

        # Entry conditions for SHORT
        entry_short = (
            (close < lower_b) &
            (rsi > 100 - params["exit_rsi_threshold"])
        )

        # Exit condition based on RSI overbought/oversold
        exit_long = rsi > params["exit_rsi_threshold"]
        exit_short = rsi < 100 - params["exit_rsi_threshold"]

        # Initialize positions
        position = 0
        entry_price = close[0]

        for i in range(n):
            if position == 0:
                if entry_long[i]:
                    position = 1
                    entry_price = close[i]
                    signals.iloc[i] = 1.0
                elif entry_short[i]:
                    position = -1
                    entry_price = close[i]
                    signals.iloc[i] = -1.0
            else:
                # Calculate stop-loss and take-profit prices
                sl_price = entry_price * (1 - params["sl_percent"] / 100)
                tp_price = entry_price * (1 + params["tp_percent"] / 100)

                current_close = close[i]

                if position == 1:
                    if current_close < sl_price or exit_long[i]:
                        position = 0
                        signals.iloc[i] = 0.0
                else:
                    if current_close > sl_price or exit_short[i]:
                        position = 0
                        signals.iloc[i] = 0.0

        return signals