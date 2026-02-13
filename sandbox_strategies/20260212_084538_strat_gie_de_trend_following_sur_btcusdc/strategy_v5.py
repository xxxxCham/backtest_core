from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_trend_following_sma_adx_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["sma", "adx", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"adx_threshold": 25, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_threshold": ParameterSpec(10, 50, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 10.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        adx_threshold = params.get("adx_threshold", 25)
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 5.0)
        warmup = int(params.get("warmup", 50))
        
        # Warmup protection
        signals.iloc[:warmup] = 0.0
        
        # Get indicator values
        sma_50 = np.nan_to_num(indicators["sma"][50])
        adx = np.nan_to_num(indicators["adx"]["adx"])
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)
        
        # Entry conditions
        entry_long = (close > sma_50) & (adx > adx_threshold) & (rsi < 30)
        
        # Exit conditions
        exit_long = (sma_50 < close) | (adx < adx_threshold) | (rsi > 70)
        
        # Generate signals
        entry_indices = np.where(entry_long)[0]
        exit_indices = np.where(exit_long)[0]
        
        for i in range(len(entry_indices)):
            entry_idx = entry_indices[i]
            # Find next exit
            next_exit = len(df) - 1
            for j in range(entry_idx + 1, len(df)):
                if j in exit_indices:
                    next_exit = j
                    break
            # Set signal for entry
            signals.iloc[entry_idx] = 1.0
            # Set signal for exit
            if next_exit > entry_idx:
                signals.iloc[next_exit] = 0.0
                
        return signals