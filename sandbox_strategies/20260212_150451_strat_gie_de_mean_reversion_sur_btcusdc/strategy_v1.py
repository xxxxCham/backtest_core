from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="donchian_rsi_mean_reversion")

    @property
    def required_indicators(self) -> List[str]:
        return ["donchian", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.0, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(50, 90, 1),
            "rsi_oversold": ParameterSpec(10, 50, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(0.5, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec(1.0, 4.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Extract indicators
        donchian = indicators["donchian"]
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Extract donchian bands
        upper_band = np.nan_to_num(donchian["upper"])
        middle_band = np.nan_to_num(donchian["middle"])
        lower_band = np.nan_to_num(donchian["lower"])
        
        # Entry conditions
        rsi_overbought = params.get("rsi_overbought", 70)
        price_touch_upper = df["close"] >= upper_band
        rsi_condition = rsi > rsi_overbought
        
        # Long entry
        long_entry = price_touch_upper & rsi_condition
        
        # Exit condition
        exit_condition = df["close"] < middle_band
        
        # Generate signals
        entry_points = np.where(long_entry, 1.0, 0.0)
        exit_points = np.where(exit_condition, -1.0, 0.0)
        
        # Combine signals
        signals = pd.Series(entry_points, index=df.index, dtype=np.float64)
        # Apply exit signals
        for i in range(len(signals)):
            if signals.iloc[i] == 1.0:
                # Look ahead for exit
                for j in range(i+1, len(signals)):
                    if exit_points[j] == -1.0:
                        signals.iloc[j] = -1.0
                        break
        
        return signals