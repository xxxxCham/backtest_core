from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="ema_rsi_bollinger")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi", "bollinger"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "bollinger_std_dev": 2,
            "ema_period": 21,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "warmup": 50
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "bollinger_std_dev": ParameterSpec(
                type=float,
                bounds=(1.0, 3.0),
                default=2.0,
                description="Number of standard deviations for Bollinger Bands"
            ),
            "ema_period": ParameterSpec(
                type=int,
                bounds=(9, 50),
                default=21,
                description="Period for EMA calculation"
            ),
            "rsi_period": ParameterSpec(
                type=int,
                bounds=(8, 21),
                default=14,
                description="Period for RSI calculation"
            ),
            "stop_atr_mult": ParameterSpec(
                type=float,
                bounds=(1.0, 3.0),
                default=1.5,
                description="Multiple of ATR for stop loss"
            ),
            "tp_atr_mult": ParameterSpec(
                type=float,
                bounds=(2.0, 4.0),
                default=3.0,
                description="Multiple of ATR for take profit"
            ),
            "warmup": ParameterSpec(
                type=int,
                bounds=(50, 100),
                default=50,
                description="Number of bars to skip for initial warmup"
            )
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Warmup period to avoid NaN signals
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Get indicators with NaN converted to 0
        ema_21 = np.nan_to_num(indicators["ema"][str(params["ema_period"])])
        rsi = np.nan_to_num(indicators["rsi"][str(params["rsi_period"])])
        bb = indicators["bollinger"]
        upper_bb = np.nan_to_num(bb["upper"])
        lower_bb = np.nan_to_num(bb["lower"])
        mid_bb = np.nan_to_num(bb["mid"])
        
        # Entry conditions
        long_entry = (
            (df.close > ema_21) &
            (rsi > 50) &
            (df.close < upper_bb)
        )
        
        short_entry = (
            (df.close < ema_21) &
            (rsi < 50) &
            (df.close > lower_bb)
        )
        
        # Exit conditions
        in_long = signals.shift(1).fillna(0).astype(int) == 1
        in_short = signals.shift(1).fillna(0).astype(int) == -1
        
        long_exit = (
            (df.close > upper_bb) |
            (rsi < 50)
        )
        
        short_exit = (
            (df.close < lower_bb) |
            (rsi > 50)
        )
        
        # Generate signals
        for i in range(warmup, len(df)):
            if long_entry[i]:
                signals.iloc[i] = 1.0
            elif short_entry[i]:
                signals.iloc[i] = -1.0
            elif in_long[i] and long_exit[i]:
                signals.iloc[i] = 0.0
            elif in_short[i] and short_exit[i]:
                signals.iloc[i] = 0.0
        
        return signals