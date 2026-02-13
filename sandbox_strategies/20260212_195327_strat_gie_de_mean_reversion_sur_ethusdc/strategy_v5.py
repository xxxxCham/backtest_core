from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="donchian_williams_r_mean_reversion")

    @property
    def required_indicators(self) -> List[str]:
        return ["donchian", "williams_r", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"donchian_period": 20, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50, "williams_r_overbought": -20, "williams_r_oversold": -80, "williams_r_period": 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "donchian_period": ParameterSpec(10, 50, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1),
            "williams_r_overbought": ParameterSpec(-30, -10, 1),
            "williams_r_oversold": ParameterSpec(-90, -70, 1),
            "williams_r_period": ParameterSpec(5, 30, 1),
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
        williams_r = indicators["williams_r"]
        atr = indicators["atr"]
        
        # Compute donchian bands
        donchian_upper = np.nan_to_num(donchian["upper"])
        donchian_middle = np.nan_to_num(donchian["middle"])
        donchian_lower = np.nan_to_num(donchian["lower"])
        
        # Compute williams_r
        williams_r_val = np.nan_to_num(williams_r)
        
        # Compute atr
        atr_val = np.nan_to_num(atr)
        
        # Entry conditions
        # Short only
        short_entry_cond = (df["close"] <= donchian_lower) & (williams_r_val <= params["williams_r_oversold"])
        
        # Exit conditions
        short_exit_cond = (df["close"] >= donchian_middle) | (williams_r_val >= params["williams_r_overbought"])
        
        # Generate signals
        entry_points = np.where(short_entry_cond, -1.0, 0.0)
        exit_points = np.where(short_exit_cond, 0.0, -1.0)
        
        # Combine entry and exit
        signals.iloc[:] = entry_points
        # Set exit signals
        signals = signals.where(~short_exit_cond, 0.0)
        
        return signals