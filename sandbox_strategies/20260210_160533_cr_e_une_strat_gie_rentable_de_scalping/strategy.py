from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    """
    Auto-generated strategy: Dogecoin ScalpReversal
    Objective: Scalping strategy for DOGE using RSI and Bollinger Bands with ADX filtering.
    """

    def __init__(self):
        super().__init__(name="Dogecoin ScalpReversal")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "adx"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "adx_period": 14,
            "bollinger_period": 20,
            "rsi_period": 14,
            "sl_multiplier": 2.0,
            "tp_multiplier": 3.0
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(int, 7, 28),
            "bollinger_period": ParameterSpec(int, 10, 40),
            "rsi_period": ParameterSpec(int, 5, 30),
            "sl_multiplier": ParameterSpec(float, 1.5, 2.5),
            "tp_multiplier": ParameterSpec(float, 2.0, 4.0)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Get required data and indicators
        close = df["close"].values
        rsi = np.nan_to_num(indicators["rsi"])
        bollinger = indicators["bollinger"]
        adx = np.nan_to_num(indicators["adx"]["adx"])

        # Get Bollinger Bands (upper, middle, lower)
        upper_band = np.nan_to_num(bollinger[0])
        middle_band = np.nan_to_num(bollinger[1])
        lower_band = np.nan_to_num(bollinger[2])

        # Calculate ATR stop-loss and take-profit levels
        atr = np.nan_to_num(indicators["atr"])
        sl_level_long = close - (atr * params["sl_multiplier"])
        tp_level_long = close + (atr * params["tp_multiplier"])
        sl_level_short = close + (atr * params["sl_multiplier"])
        tp_level_short = close - (atr * params["tp_multiplier"])

        # Initialize positions
        position = 0.0
        
        for i in range(1, n):
            # Entry conditions
            if (close[i] <= lower_band[i]) and \
               (rsi[i] < 30) and \
               (adx[i] < 25):
                position = 1.0

            elif (close[i] >= upper_band[i]) and \
                 (rsi[i] > 70) and \
                 (adx[i] < 25):
                position = -1.0

            # Exit conditions
            if position == 1.0:
                if close[i] <= sl_level_long[i]:
                    position = 0.0
                elif close[i] >= tp_level_long[i]:
                    position = 0.0
            
            elif position == -1.0:
                if close[i] >= sl_level_short[i]:
                    position = 0.0
                elif close[i] <= tp_level_short[i]:
                    position = 0.0

            signals.iloc[i] = position

        return signals