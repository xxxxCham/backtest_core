from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    # Dogecoin Mean Reversion with ADX Filter strategy
    def __init__(self):
        super().__init__(name="Dogecoin Mean Reversion with ADX Filter")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "adx_threshold": 25,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "sl_multiplier": 2.0,
            "tp_multiplier": 3.0
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_threshold": ParameterSpec(
                name="ADX Threshold",
                type=float,
                min=10,
                max=50,
                default=25
            ),
            "rsi_overbought": ParameterSpec(
                name="RSI Overbought Level",
                type=int,
                min=50,
                max=90,
                default=70
            ),
            "rsi_oversold": ParameterSpec(
                name="RSI Oversold Level", 
                type=int,
                min=10,
                max=50,
                default=30
            ),
            "sl_multiplier": ParameterSpec(
                name="Stop Loss Multiplier",
                type=float,
                min=1.0,
                max=3.0,
                default=2.0
            ),
            "tp_multiplier": ParameterSpec(
                name="Take Profit Multiplier", 
                type=float,
                min=2.0,
                max=4.0,
                default=3.0
            )
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any]
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Get indicators
        rsi = np.nan_to_num(indicators["rsi"])
        bollinger = indicators["bollinger"]
        adx_val = np.nan_to_num(indicators["adx"]["adx"])
        atr_val = np.nan_to_num(indicators["atr"])

        # Get price data
        close_prices = df["close"].values

        # Calculate Bollinger Bands
        lower_band = np.nan_to_num(bollinger["lower"])
        upper_band = np.nan_to_num(bollinger["upper"])

        # Filter for ADX (weak trend)
        adx_filter = adx_val < params.get("adx_threshold", 25)

        # Long entry conditions
        long_entry = (
            close_prices <= lower_band 
            & rsi < params.get("rsi_oversold", 30) 
            & adx_filter
        )

        # Short entry conditions
        short_entry = (
            close_prices >= upper_band 
            & rsi > params.get("rsi_overbought", 70)
            & adx_filter
        )

        # Set signals
        signals[long_entry] = 1.0
        signals[short_entry] = -1.0

        return signals