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
            "rsi_overbought": ParameterSpec(60, 90, 1),
            "rsi_oversold": ParameterSpec(10, 40, 1),
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
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_period = params.get("rsi_period", 14)
        stop_atr_mult = params.get("stop_atr_mult", 1.0)
        tp_atr_mult = params.get("tp_atr_mult", 2.0)
        
        donchian = indicators["donchian"]
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])
        
        lower_band = np.nan_to_num(donchian["lower"])
        middle_band = np.nan_to_num(donchian["middle"])
        upper_band = np.nan_to_num(donchian["upper"])
        
        # volatility filter
        atr_mean = np.mean(atr)
        volatility_filter = atr > atr_mean * 1.2
        
        # entry conditions
        entry_condition = (df["close"] <= lower_band) & (rsi < rsi_oversold) & volatility_filter
        
        # exit condition
        exit_condition = df["close"] > middle_band
        
        # generate signals
        long_entries = entry_condition
        long_exits = exit_condition
        
        # Initialize signals to 0.0
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Set long signals
        signals[long_entries] = 1.0
        
        # Set flat signals on exits
        signals[long_exits] = 0.0
        
        # Set warmup period to 0.0
        signals.iloc[:warmup] = 0.0
        
        return signals