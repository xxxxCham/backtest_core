from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    # Auto-generated strategy: ContinuationScalpStrategy
    # Objective: Capture short-term continuations or micro-reversals using EMA, RSI, and Bollinger Bands
    
    def __init__(self):
        super().__init__(name="ContinuationScalpStrategy")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi", "bollinger"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "minimum_risk_reward_ratio": 1.5,
            "risk_percentage_per_trade": 1,
            "stop_loss_multiplier": 1,
            "ema_period": 21,
            "rsi_overbought": 70,
            "rsi_oversold": 30
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "minimum_risk_reward_ratio": ParameterSpec(
                name="Minimum Risk/Reward Ratio",
                current_value=1.5,
                min_value=1.0,
                max_value=3.0,
                type=float,
                tunable=True
            ),
            "risk_percentage_per_trade": ParameterSpec(
                name="Risk Percentage Per Trade",
                current_value=1,
                min_value=0.5,
                max_value=2,
                type=float,
                tunable=True
            ),
            "stop_loss_multiplier": ParameterSpec(
                name="Stop Loss Multiplier",
                current_value=1,
                min_value=0.5,
                max_value=2,
                type=float,
                tunable=True
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
        
        # Get required data
        close = np.array(df["close"], dtype=np.float64)
        high = np.array(df["high"], dtype=np.float64)
        low = np.array(df["low"], dtype=np.float64)

        # Access indicators
        ema21 = np.nan_to_num(indicators["ema"][params.get("ema_period", 21)])
        rsi = np.nan_to_num(indicators["rsi"])
        
        bb = indicators["bollinger"]
        upper_bb = np.nan_to_num(bb["upper"])
        lower_bb = np.nan_to_num(bb["lower"])

        # Initialize entry conditions
        for i in range(1, n):
            if signals[i-1] == 0:
                # Look for new entries
                # LONG condition
                if (
                    close[i] > ema21[i] and 
                    rsi[i] > params.get("rsi_oversold", 30) and 
                    (close[i-1] < upper_bb[i-1] or low[i] < upper_bb[i])
                ):
                    signals[i] = 1.0
                
                # SHORT condition
                elif (
                    close[i] < ema21[i] and 
                    rsi[i] < params.get("rsi_overbought", 70) and 
                    (close[i-1] > lower_bb[i-1] or high[i] > lower_bb[i])
                ):
                    signals[i] = -1.0
            
            else:
                # Check exit conditions
                current_position = signals[i-1]
                
                if current_position == 1.0:
                    # Exit LONG
                    if close[i] >= upper_bb[i]:
                        signals[i] = 0.0
                    elif rsi[i] < rsi[i-1] and high[i] > high[i-1]:
                        signals[i] = 0.0
                
                elif current_position == -1.0:
                    # Exit SHORT
                    if close[i] <= lower_bb[i]:
                        signals[i] = 0.0
                    elif rsi[i] > rsi[i-1] and low[i] < low[i-1]:
                        signals[i] = 0.0

        return signals