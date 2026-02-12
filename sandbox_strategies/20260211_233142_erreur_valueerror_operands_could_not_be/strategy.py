from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_optimized")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(),
            "rsi_oversold": ParameterSpec(),
            "rsi_period": ParameterSpec(),
            "stop_atr_mult": ParameterSpec(),
            "tp_atr_mult": ParameterSpec(),
            "warmup": ParameterSpec(),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract required data
        close_price = np.array(df["close"])
        rsi_values = np.nan_to_num(indicators["rsi"])
        bollinger_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bollinger_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        atr_values = np.nan_to_num(indicators["atr"])

        # Calculate entry conditions
        long_entry_condition = close_price > bollinger_upper * 0.98
        long_entry_condition &= rsi_values < params["rsi_oversold"]
        
        short_entry_condition = close_price < bollinger_lower * 1.02
        short_entry_condition &= rsi_values > params["rsi_overbought"]

        # Calculate stop loss and take profit levels
        stop_loss_long = close_price - atr_values * params["stop_atr_mult"]
        take_profit_long = close_price + atr_values * params["tp_atr_mult"]
        
        stop_loss_short = close_price + atr_values * params["stop_atr_mult"]
        take_profit_short = close_price - atr_values * params["tp_atr_mult"]

        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Assign signals
        long_entry = np.where(long_entry_condition, 1.0, 0.0)
        short_entry = np.where(short_entry_condition, -1.0, 0.0)

        signals[:] = np.where(
            (long_entry == 1) | (short_entry == -1),
            long_entry + short_entry,
            0.0
        )

        return signals