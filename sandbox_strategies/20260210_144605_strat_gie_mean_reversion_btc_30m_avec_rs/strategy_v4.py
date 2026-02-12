from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    """
    Auto-generated strategy: custom
    Objective: Stratégie mean-reversion BTC 30m avec RSI + Stochastic + Bollinger. Ajuster les seuils overbought/oversold et les périodes à chaque itération.
    Indicators: 
    """

    def __init__(self):
        super().__init__(name="BuilderStrategy")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "stochastic", "bollinger"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "stoch_k_overbought": 80,
            "stoch_k_oversold": 20,
            "bollinger_upper_threshold": 2,
            "bollinger_lower_threshold": -2,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(
                name="RSI Overbought",
                type="int",
                default=70,
                min=40,
                max=90,
                step=1,
                description="RSI threshold for overbought condition",
            ),
            "rsi_oversold": ParameterSpec(
                name="RSI Oversold",
                type="int",
                default=30,
                min=0,
                max=60,
                step=1,
                description="RSI threshold for oversold condition",
            ),
            "stoch_k_overbought": ParameterSpec(
                name="Stochastic %k Overbought",
                type="float",
                default=80.0,
                min=50.0,
                max=100.0,
                step=1.0,
                description="Stochastic %k threshold for overbought condition",
            ),
            "stoch_k_oversold": ParameterSpec(
                name="Stochastic %k Oversold",
                type="float",
                default=20.0,
                min=0.0,
                max=50.0,
                step=1.0,
                description="Stochastic %k threshold for oversold condition",
            ),
            "bollinger_upper_threshold": ParameterSpec(
                name="Bollinger Upper Threshold",
                type="float",
                default=2.0,
                min=1.0,
                max=5.0,
                step=0.5,
                description="Number of standard deviations for Bollinger upper band",
            ),
            "bollinger_lower_threshold": ParameterSpec(
                name="Bollinger Lower Threshold",
                type="float",
                default=-2.0,
                min=-5.0,
                max=-1.0,
                step=0.5,
                description="Number of standard deviations for Bollinger lower band",
            ),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        # Get the indicators
        rsi = indicators["rsi"]
        stochastic_k, stochastic_d = indicators["stochastic"]
        upper, middle, lower = indicators["bollinger"]

        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Unpack the parameters
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        stoch_k_overbought = params["stoch_k_overbought"]
        stoch_k_oversold = params["stoch_k_oversold"]
        bollinger_upper_threshold = params["bollinger_upper_threshold"]
        bollinger_lower_threshold = params["bollinger_lower_threshold"]

        # Create masks for conditions
        # LONG condition: RSI oversold, Stoch k oversold, and price below Bollinger lower band
        long_condition = (
            (rsi < rsi_oversold)
            & (stochastic_k < stoch_k_oversold)
            & (df["close"] < lower)
            & (df["close"] > middle - 2 * (middle - lower))  # Ensure not too far from middle band
        )

        # SHORT condition: RSI overbought, Stoch k overbought, and price above Bollinger upper band
        short_condition = (
            (rsi > rsi_overbought)
            & (stochastic_k > stoch_k_overbought)
            & (df["close"] > upper)
            & (df["close"] < middle + 2 * (upper - middle))  # Ensure not too far from middle band
        )

        # Apply signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0

        # Replace NaN with 0 (no signal)
        signals = signals.fillna(0.0)

        return signals