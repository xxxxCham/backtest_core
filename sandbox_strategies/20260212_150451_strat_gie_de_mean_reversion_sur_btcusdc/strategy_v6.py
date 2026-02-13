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
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(10, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        donchian = indicators["donchian"]
        rsi = indicators["rsi"]
        atr = indicators["atr"]
        
        # Get Donchian bands
        upper_band = np.nan_to_num(donchian["upper"])
        lower_band = np.nan_to_num(donchian["lower"])
        middle_band = np.nan_to_num(donchian["middle"])
        
        # Get RSI values
        rsi = np.nan_to_num(rsi)
        
        # Get ATR values
        atr = np.nan_to_num(atr)
        
        # Get parameters
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))
        
        # Initialize entry conditions
        entry_long = (df["close"] <= lower_band) & (rsi < rsi_oversold) & (df["close"].shift(1) > np.roll(lower_band, 1))
        
        # Initialize exit condition
        exit_long = df["close"] > middle_band
        
        # Set signals
        entry_indices = np.where(entry_long)[0]
        exit_indices = np.where(exit_long)[0]
        
        # Set signals for entry and exit
        for idx in entry_indices:
            if idx >= warmup:
                signals.iloc[idx] = 1.0  # Long signal
        
        # Set exit signals
        for idx in exit_indices:
            if idx >= warmup and signals.iloc[idx] == 1.0:
                signals.iloc[idx] = 0.0  # Flat signal
        
        # Set warmup period
        signals.iloc[:warmup] = 0.0
        
        return signals