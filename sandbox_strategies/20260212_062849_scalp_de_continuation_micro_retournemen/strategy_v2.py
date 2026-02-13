from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="scaly_snake_case_name")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "bollinger_period": 20,
            "bollinger_std_dev": 2,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "warmup": 50,
            "risk percent": 1.5,
            "risk_ratio_min": 1.5
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "bollinger_period": ParameterSpec(type_=(int, float), min_val=10, max_val=100),
            "bollinger_std_dev": ParameterSpec(type_=(int, float), min_val=1, max_val=5),
            "rsi_overbought": ParameterSpec(type_=(int, float), min_val=50, max_val=90),
            "rsi_oversold": ParameterSpec(type_=(int, float), min_val=10, max_val=50),
            "rsi_period": ParameterSpec(type_=(int, float), min_val=7, max_val=21),
            "stop_atr_mult": ParameterSpec(type_=(int, float), min_val=1.0, max_val=3.0),
            "tp_atr_mult": ParameterSpec(type_=(int, float), min_val=1.0, max_val=5.0),
            "warmup": ParameterSpec(type_=(int, float), min_val=0, max_val=100),
            "risk_percent": ParameterSpec(type_=(int, float), min_val=0.5, max_val=3.0),
            "risk_ratio_min": ParameterSpec(type_=(int, float), min_val=1.0, max_val=3.0),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Extract and sanitize indicator arrays
        bollinger = indicators["bollinger"]
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Get parameter values
        lower_band = np.nan_to_num(bollinger["lower"])
        upper_band = np.nan_to_num(bollinger["upper"])
        risk_percent_val = float(params.get("risk_percent", 1.5))
        
        # Entry conditions
        long_entry = (
            (rsi > params["rsi_oversold"]) &
            (rsi < params["rsi_overbought"]) &
            (df["close"] <= upper_band) &
            (df["close"] >= lower_band)
        )
        
        short_entry = (
            (rsi > params["rsi_oversold"]) &
            (rsi < params["rsi_overbought"]) &
            (df["close"] <= upper_band) &
            (df["close"] >= lower_band)
        )
        
        # Exit conditions
        long_exit = (
            (rsi > params["rsi_oversold"]) |
            (rsi < params["rsi_oversold"])
        ) & (
            (df["close"] > lower_band) |
            (df["close"] < upper_band)
        )
        
        short_exit = (
            (rsi < params["rsi_oversold"]) |
            (rsi > params["rsi_oversold"])
        ) & (
            (df["close"] < upper_band) |
            (df["close"] > lower_band)
        )
        
        # Risk management
        entry_cost = df["close"].shift(1)
        stop_price_long = entry_cost - (entry_cost * params["stop_atr_mult"] * atr / 100)
        stop_price_short = entry_cost + (entry_cost * params["stop_atr_mult"] * atr / 100)
        tp_price_long = entry_cost + (entry_cost * params["tp_atr_mult"] * atr / 100)
        tp_price_short = entry_cost - (entry_cost * params["tp_atr_mult"] * atr / 100)
        
        # Apply signals
        long_condition = long_entry & ~np.roll(long_entry, 1) & long_exit
        short_condition = short_entry & ~np.roll(short_entry, 1) & short_exit
        
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0
        
        return signals