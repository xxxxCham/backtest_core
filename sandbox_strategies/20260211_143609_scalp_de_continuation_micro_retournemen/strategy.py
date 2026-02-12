from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    # Auto-generated strategy: ScalpContinuationBandsV2
    # Objective: Scalp de continuation/micro-retournement using EMA, RSI, and Bollinger Bands
    
    def __init__(self):
        super().__init__(name="ScalpContinuationBandsV2")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "bollinger_period": 20,
            "bollinger_std_dev": 2,
            "ema_periods": [21, 50],
            "rsi_period": 14,
            "stop_loss_mult": 2,
            "take_profit_mult": 1.5
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "bollinger_period": ParameterSpec(type=float, min=10, max=30),
            "ema_periods": ParameterSpec(type=list, default=[21, 50]),
            "rsi_period": ParameterSpec(type=int, min=10, max=30),
            "stop_loss_mult": ParameterSpec(type=float, min=1, max=3),
            "take_profit_mult": ParameterSpec(type=float, min=1, max=2)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Get precomputed indicators
        ema_indicator = indicators["ema"]
        rsi_val = np.nan_to_num(indicators["rsi"])
        bollinger = indicators["bollinger"]
        
        # Extract EMA values for periods 21 and 50
        ema_21 = np.nan_to_num(ema_indicator[params.get("ema_periods")[0]])
        ema_50 = np.nan_to_num(ema_indicator[params.get("ema_periods")[1]])
        
        # Bollinger Bands values
        upper_band = np.nan_to_num(bollinger["upper"])
        lower_band = np.nan_to_num(bollinger["lower"])

        # Entry conditions
        long_entry = (
            (ema_21 > ema_50) &
            (df["close"].values > lower_band) &
            (rsi_val < params.get("long_rsi_threshold", 35))
        )
        
        short_entry = (
            (ema_21 < ema_50) &
            (df["close"].values < upper_band) &
            (rsi_val > params.get("short_rsi_threshold", 65))
        )

        # Exit conditions
        exit_long = df["close"].values >= upper_band
        exit_short = df["close"].values <= lower_band

        for i in range(n):
            if i < max(params.get("ema_periods")):
                signals[i] = 0.0
                continue
                
            current_close = df["close"].values[i]
            
            # Check entry conditions
            if long_entry[i]:
                signals[i] = 1.0
            elif short_entry[i]:
                signals[i] = -1.0
            
            # Check exit conditions
            if signals[i-1] == 1.0 and (exit_long[i] or current_close <= lower_band[i]):
                signals[i] = 0.0
            elif signals[i-1] == -1.0 and (exit_short[i] or current_close >= upper_band[i]):
                signals[i] = 0.0

        return signals