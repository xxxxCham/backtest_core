from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    """
    Auto-generated strategy: Dogecoin Mean Reversion Scalper
    Objective: Generate profitable trades by capturing mean-reverting price movements in Dogecoin using precise entry conditions based on RSI, Bollinger Bands, and Vortex. The use of ATR-based stop-loss and take-profit with ADX filtering should control risk and reduce overtrading.
    Indicators: rsi, bollinger, vortex
    """

    def __init__(self):
        super().__init__(name="Dogecoin Mean Reversion Scalper")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "vortex"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "leverage": 1,
            "stop_loss_multiplier": 2.0,
            "take_profit_multiplier": 3.0,
            "risk_per_trade": 0.02,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "leverage": ParameterSpec(
                name="Leverage",
                type=float,
                min_value=1.0,
                max_value=2.0,
                default=1.0,
                step=0.5,
            ),
            "stop_loss_multiplier": ParameterSpec(
                name="Stop Loss Multiplier",
                type=float,
                min_value=1.5,
                max_value=2.5,
                default=2.0,
                step=0.5,
            ),
            "take_profit_multiplier": ParameterSpec(
                name="Take Profit Multiplier",
                type=float,
                min_value=2.0,
                max_value=4.0,
                default=3.0,
                step=0.5,
            ),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Get required data
        close_prices = df["close"].values
        high_prices = df["high"].values
        low_prices = df["low"].values

        # Get indicators
        rsi = np.nan_to_num(indicators["rsi"])
        bollinger = indicators["bollinger"]
        lower_bollinger = np.nan_to_num(bollinger["lower"])
        upper_bollinger = np.nan_to_num(bollinger["upper"])

        vortex = indicators["vortex"]
        vi_plus = np.nan_to_num(vortex["vi_plus"])
        vi_minus = np.nan_to_num(vortex["vi_minus"])

        # Initialize variables
        position = 0.0
        stop_loss_level = 0.0
        take_profit_level = 0.0

        for i in range(1, n):
            current_close = close_prices[i]
            prev_close = close_prices[i-1]

            # Calculate ATR-based levels
            atr = np.nan_to_num(indicators["atr"][i])
            stop_loss = params.get("stop_loss_multiplier", 2.0) * atr
            take_profit = params.get("take_profit_multiplier", 3.0) * atr

            # Entry conditions
            if position == 0:
                # LONG entry
                if (
                    rsi[i] < 40 and 
                    current_close < lower_bollinger[i] and
                    vi_plus[i] > vi_minus[i]
                ):
                    position = 1.0
                    stop_loss_level = prev_close - stop_loss
                    take_profit_level = prev_close + take_profit

                # SHORT entry
                elif (
                    rsi[i] > 60 and 
                    current_close > upper_bollinger[i] and
                    vi_minus[i] > vi_plus[i]
                ):
                    position = -1.0
                    stop_loss_level = prev_close + stop_loss
                    take_profit_level = prev_close - take_profit

            # Exit conditions
            else:
                if (position == 1.0 and 
                    (current_close <= stop_loss_level or current_close >= take_profit_level)):
                    position = 0.0
                elif (position == -1.0 and 
                      (current_close >= stop_loss_level or current_close <= take_profit_level)):
                    position = 0.0

            signals[i] = position

        return signals