from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    # Dogecoin Scalp Reversal strategy implementation
    def __init__(self):
        super().__init__(name="Dogecoin Scalp Reversal")

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
            "adx_threshold": ParameterSpec(min=20, max=30, default=25),
            "rsi_overbought": ParameterSpec(min=60, max=80, default=70),
            "rsi_oversold": ParameterSpec(min=20, max=40, default=30),
            "sl_multiplier": ParameterSpec(min=1.5, max=2.5, default=2.0),
            "tp_multiplier": ParameterSpec(min=2.0, max=4.0, default=3.0)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any]
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Extract and clean indicator data
        close_price = df["close"].values.astype(np.float64)
        
        # Bollinger Bands
        bb = indicators["bollinger"]
        lower_bb = np.nan_to_num(bb["lower"])
        upper_bb = np.nan_to_num(bb["upper"])

        # RSI
        rsi = np.nan_to_num(indicators["rsi"])
        oversold = rsi < params.get("rsi_oversold", 30)
        overbought = rsi > params.get("rsi_overbought", 70)

        # ADX
        adx_data = indicators["adx"]
        adx_val = np.nan_to_num(adx_data["adx"])
        low_adx = adx_val < params.get("adx_threshold", 25)

        # Entry conditions
        long_entry = (
            (close_price > lower_bb) & 
            oversold &
            low_adx
        )
        
        short_entry = (
            (close_price < upper_bb) &
            overbought &
            low_adx
        )

        # Construct signals array
        for i in range(n):
            if long_entry[i]:
                signals.iloc[i] = 1.0
            elif short_entry[i]:
                signals.iloc[i] = -1.0
            else:
                signals.iloc[i] = 0.0

        return signals