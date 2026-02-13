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
        return {"donchian_period": 20, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50, "williams_r_overbought": -20, "williams_r_oversold": -80, "williams_r_period": 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "donchian_period": ParameterSpec(10, 50, 1, "Donchian period"),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1, "Stop-loss ATR multiplier"),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1, "Take-profit ATR multiplier"),
            "warmup": ParameterSpec(30, 100, 1, "Warmup bars"),
            "williams_r_overbought": ParameterSpec(-30, -50, -1, "Williams %R overbought level"),
            "williams_r_oversold": ParameterSpec(-90, -70, -1, "Williams %R oversold level"),
            "williams_r_period": ParameterSpec(5, 30, 1, "Williams %R period"),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        donchian_period = int(params.get("donchian_period", 20))
        stop_atr_mult = float(params.get("stop_atr_mult", 1.5))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.0))
        warmup = int(params.get("warmup", 50))
        williams_r_overbought = float(params.get("williams_r_overbought", -20))
        williams_r_oversold = float(params.get("williams_r_oversold", -80))
        
        # Extract indicators
        donchian = indicators["donchian"]
        upper_band = np.nan_to_num(donchian["upper"])
        middle_band = np.nan_to_num(donchian["middle"])
        lower_band = np.nan_to_num(donchian["lower"])
        
        williams_r = np.nan_to_num(indicators["williams_r"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Entry condition: price touches upper band with overbought Williams %R
        entry_condition = (df["close"] >= upper_band * 0.999) & (williams_r < williams_r_oversold)
        
        # Exit condition: price returns to middle band or enters overbought
        exit_condition = (df["close"] <= middle_band) | (williams_r > williams_r_overbought)
        
        # Generate signals
        entry_signals = entry_condition.astype(int)
        exit_signals = exit_condition.astype(int)
        
        # Long positions only
        long_positions = np.zeros_like(entry_signals)
        position = 0
        
        for i in range(len(entry_signals)):
            if position == 0 and entry_signals[i] == 1:
                position = 1
                long_positions[i] = 1
            elif position == 1 and exit_signals[i] == 1:
                position = 0
                long_positions[i] = 0
        
        signals = pd.Series(long_positions, index=df.index, dtype=np.float64)
        
        # Warmup protection
        signals.iloc[:warmup] = 0.0
        
        return signals