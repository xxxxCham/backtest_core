from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    # Dogecoin Mean Reversion with Trend Confirmation strategy
    
    def __init__(self):
        super().__init__(name="Dogecoin Mean Reversion with Trend Confirmation")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "supertrend", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "rsi_overbought": 75,
            "rsi_oversold": 25,
            "sl_multiplier": 2.5,
            "tp_multiplier": 4.0
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(70, 80, 75),
            "rsi_oversold": ParameterSpec(20, 30, 25),
            "sl_multiplier": ParameterSpec(1.5, 3.0, 2.5),
            "tp_multiplier": ParameterSpec(2.0, 4.0, 4.0)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Get indicators
        rsi = np.nan_to_num(indicators["rsi"])
        bollinger = indicators["bollinger"]
        lower_bb = np.nan_to_num(bollinger["lower"])
        upper_bb = np.nan_to_num(bollinger["upper"])
        
        supertrend = indicators["supertrend"]
        st_direction = supertrend["direction"]  # 1=up, -1=down
        
        close_prices = df["close"].values

        current_position = 0.0  # 0: flat, 1: long, -1: short

        for i in range(1, n):
            price = close_prices[i]
            
            # Check if we need to enter a position
            if current_position == 0:
                # Long entry conditions
                if (price > lower_bb[i] and 
                    rsi[i] < params["rsi_oversold"] and 
                    st_direction[i] > 0):
                    signals.iloc[i] = 1.0
                    current_position = 1.0
                
                # Short entry conditions
                elif (price < upper_bb[i] and 
                      rsi[i] > params["rsi_overbought"] and 
                      st_direction[i] < 0):
                    signals.iloc[i] = -1.0
                    current_position = -1.0
            
            else:
                # Stay in position until exit conditions met
                pass

        return signals