from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="Dogecoin Mean-Reversion Scalper")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr", "adx"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "adx_period": 14,
            "adx_threshold": 20,
            "atr_period": 14,
            "bollinger_period": 20,
            "rsi_overbought": 65,
            "rsi_oversold": 35,
            "rsi_period": 14,
            "stop_atr_mult": 2.0,
            "tp_atr_mult": 3.0,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_threshold": ParameterSpec(
                name="ADX Threshold",
                description="Minimum ADX value to confirm trend strength.",
                min_value=10,
                max_value=40,
                default=20,
                type="int",
            ),
            "rsi_overbought": ParameterSpec(
                name="RSI Overbought Level",
                description="Level above which price is considered overbought.",
                min_value=50,
                max_value=80,
                default=65,
                type="int",
            ),
            "rsi_oversold": ParameterSpec(
                name="RSI Oversold Level",
                description="Level below which price is considered oversold.",
                min_value=20,
                max_value=50,
                default=35,
                type="int",
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

        # Get required indicator values
        close_prices = np.array(df["close"])
        bollinger = indicators["bollinger"]
        lower_band = np.nan_to_num(bollinger["lower"])
        upper_band = np.nan_to_num(bollinger["upper"])
        
        rsi_val = np.nan_to_num(indicators["rsi"])
        adx_val = np.nan_to_num(indicators["adx"]["adx"])
        atr_val = np.nan_to_num(indicators["atr"])

        # Generate signals
        for i in range(1, n):
            price_prev = close_prices[i-1]
            price_curr = close_prices[i]

            # Check ADX condition first to filter weak trends
            if adx_val[i] < params.get("adx_threshold", 20):
                # Check LONG entry conditions
                if (price_curr > lower_band[i] and 
                    rsi_val[i] < params.get("rsi_oversold", 35) and
                    signals.iloc[i-1] != 1.0):  # Only enter if not already long
                    signals.iloc[i] = 1.0

                # Check SHORT entry conditions
                elif (price_curr < upper_band[i] and 
                      rsi_val[i] > params.get("rsi_overbought", 65) and
                      signals.iloc[i-1] != -1.0):  # Only enter if not already short
                    signals.iloc[i] = -1.0

            # Close LONG positions if price crosses below lower band
            if signals.iloc[i-1] == 1.0 and price_curr < lower_band[i]:
                signals.iloc[i] = 0.0

            # Close SHORT positions if price crosses above upper band
            elif signals.iloc[i-1] == -1.0 and price_curr > upper_band[i]:
                signals.iloc[i] = 0.0

        return signals