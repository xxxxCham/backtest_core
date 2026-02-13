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
            "donchian_period": ParameterSpec("donchian_period", 10, 50, 1),
            "stop_atr_mult": ParameterSpec("stop_atr_mult", 1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec("tp_atr_mult", 2.0, 5.0, 0.1),
            "warmup": ParameterSpec("warmup", 30, 100, 1),
            "williams_r_overbought": ParameterSpec("williams_r_overbought", -30, -10, 1),
            "williams_r_oversold": ParameterSpec("williams_r_oversold", -90, -70, 1),
            "williams_r_period": ParameterSpec("williams_r_period", 5, 30, 1),
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
        
        donchian_period = int(params.get("donchian_period", 20))
        stop_atr_mult = float(params.get("stop_atr_mult", 1.5))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.0))
        williams_r_overbought = float(params.get("williams_r_overbought", -20))
        williams_r_oversold = float(params.get("williams_r_oversold", -80))
        
        donchian = indicators["donchian"]
        williams_r = indicators["williams_r"]
        atr = indicators["atr"]
        
        upper = np.nan_to_num(donchian["upper"])
        middle = np.nan_to_num(donchian["middle"])
        lower = np.nan_to_num(donchian["lower"])
        wr = np.nan_to_num(williams_r)
        atr_values = np.nan_to_num(atr)
        
        price = np.nan_to_num(df["close"].values)
        
        # Short entry conditions
        entry_condition = (price == upper) & (wr <= williams_r_oversold)
        
        # Exit conditions
        exit_condition = (price <= middle) | (wr >= williams_r_overbought)
        
        # Generate signals
        entry_mask = entry_condition
        exit_mask = exit_condition
        
        # Initialize signal array
        signal_values = np.zeros_like(price)
        
        # Set short signals
        signal_values[entry_mask] = -1.0
        
        # Set flat signals when exit condition met
        signal_values[exit_mask] = 0.0
        
        # Create final signals series
        signals = pd.Series(signal_values, index=df.index, dtype=np.float64)
        
        return signals