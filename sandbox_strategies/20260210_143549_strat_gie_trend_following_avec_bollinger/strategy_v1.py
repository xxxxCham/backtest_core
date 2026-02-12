from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    """
    Auto-generated strategy: BollingerRSITrendFollowing
    Objective: Stratégie trend-following avec Bollinger + RSI
    Indicators: bollinger, rsi
    """

    def __init__(self):
        super().__init__(name="BollingerRSITrendFollowing")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"bollinger_period": 20, "rsi_period": 14, "standard_deviations": 2}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "bollinger_period": ParameterSpec(
                type_=int,
                default=20,
                bounds=(2, 100),
                description="Period for Bollinger Bands calculation"
            ),
            "rsi_period": ParameterSpec(
                type_=int,
                default=14,
                bounds=(2, 100),
                description="Period for RSI calculation"
            ),
            "standard_deviations": ParameterSpec(
                type_=float,
                default=2.0,
                bounds=(1.0, 4.0),
                description="Number of standard deviations for Bollinger Bands"
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
        bollinger_upper, _, bollinger_lower = indicators["bollinger"]
        
        # Extract RSI
        rsi = indicators["rsi"]

        # Get close prices
        close_prices = df['close'].values

        # Initialize positions and states
        position = 0  # Can be -1 (SHORT), 0 (FLAT), or 1 (LONG)
        prev_rsi = None
        
        for i in range(n):
            current_close = close_prices[i]
            
            # Check if we have valid data for indicators
            if np.isnan(current_close) or np.isnan(rsi[i]) or np.isnan(bollinger_upper[i]) or np.isnan(bollinger_lower[i]):
                signals[i] = 0.0
                continue
                
            # Entry conditions
            if position == 0:
                # Entry LONG: Price closes above upper Bollinger Band AND RSI is below 30 (oversold)
                entry_long = current_close > bollinger_upper[i] and rsi[i] < 30
                # Entry SHORT: Price closes below lower Bollinger Band AND RSI is above 70 (overbought)
                entry_short = current_close < bollinger_lower[i] and rsi[i] > 70
                
                if entry_long:
                    position = 1
                    signals[i] = 1.0
                elif entry_short:
                    position = -1
                    signals[i] = -1.0
                else:
                    signals[i] = 0.0
                    
            else:
                # Exit conditions based on RSI crossing back or trend reversal
                if prev_rsi is not None:
                    if (position == 1 and rsi[i] >= prev_rsi) or (position == -1 and rsi[i] <= prev_rsi):
                        position = 0
                        signals[i] = 0.0
                        
                # Check for trend reversal based on Bollinger Bands
                if position == 1 and current_close < bollinger_upper[i]:
                    position = 0
                    signals[i] = 0.0
                elif position == -1 and current_close > bollinger_lower[i]:
                    position = 0
                    signals[i] = 0.0
                    
            prev_rsi = rsi[i]
            
        return signals