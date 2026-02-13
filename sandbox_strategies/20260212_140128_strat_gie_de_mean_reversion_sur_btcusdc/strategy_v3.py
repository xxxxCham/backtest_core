from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="donchian_williams_mean_reversion")

    @property
    def required_indicators(self) -> List[str]:
        return ["donchian", "williams_r", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"donchian_period": 20, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50, "williams_r_overbought": -20, "williams_r_oversold": -80}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "donchian_period": ParameterSpec(10, 50, 1, "Donchian period"),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1, "Stop-loss ATR multiplier"),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1, "Take-profit ATR multiplier"),
            "williams_r_overbought": ParameterSpec(-50, -10, 1, "Williams %R overbought level"),
            "williams_r_oversold": ParameterSpec(-90, -50, 1, "Williams %R oversold level"),
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
        upper_band = np.nan_to_num(donchian["upper"])
        middle_band = np.nan_to_num(donchian["middle"])
        lower_band = np.nan_to_num(donchian["lower"])
        
        williams_r = np.nan_to_num(indicators["williams_r"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Entry condition: price approaches upper band and Williams %R is oversold
        entry_condition = (df["close"] >= upper_band * 0.99) & (williams_r < williams_r_oversold)
        
        # Exit condition: price returns to middle band or Williams %R moves to overbought
        exit_condition = (df["close"] <= middle_band) | (williams_r > williams_r_overbought)
        
        # Generate long signals
        long_entries = entry_condition
        long_exits = exit_condition
        
        # Create signal array
        entry_indices = np.where(long_entries)[0]
        exit_indices = np.where(long_exits)[0]
        
        # Initialize signals
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Set initial signals
        signals.iloc[:warmup] = 0.0
        
        # Apply long signals
        for i in entry_indices:
            if i >= warmup:
                signals.iloc[i] = 1.0  # Long signal
                
                # Set exit at next bar after entry
                if i + 1 < len(signals):
                    signals.iloc[i + 1] = 0.0  # Flat signal
                    
        return signals