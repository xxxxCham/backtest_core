from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_momentum_short_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "mfi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 2.0, "tp_atr_mult": 4.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(20, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(3.0, 6.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        rsi = np.nan_to_num(indicators["rsi"])
        mfi = np.nan_to_num(indicators["mfi"])
        atr = np.nan_to_num(indicators["atr"])
        
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 4.5)
        warmup = int(params.get("warmup", 50))
        
        # Short entry: RSI crosses below 30 and MFI is above 50
        rsi_below_oversold = rsi < rsi_oversold
        mfi_above_50 = mfi > 50
        
        # Entry condition
        entry_condition = rsi_below_oversold & mfi_above_50
        
        # Previous RSI values for crossover detection
        rsi_shifted = np.roll(rsi, 1)
        rsi_crossed_below = (rsi_shifted < rsi_oversold) & (rsi >= rsi_oversold)
        
        # Final entry signal
        entry_signal = entry_condition & rsi_crossed_below
        
        # Set signals
        signals[entry_signal] = -1.0
        
        # Warmup protection
        signals.iloc[:warmup] = 0.0
        
        return signals