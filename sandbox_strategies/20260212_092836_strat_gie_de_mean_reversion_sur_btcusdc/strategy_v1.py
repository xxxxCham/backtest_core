from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btc_bollinger_rsi_mean_reversion")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 90, 1),
            "rsi_oversold": ParameterSpec(10, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.5),
            "tp_atr_mult": ParameterSpec(2.0, 10.0, 0.5),
            "warmup": ParameterSpec(20, 100, 5),
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
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        lower_band = np.nan_to_num(bb["lower"])
        middle_band = np.nan_to_num(bb["middle"])
        close = np.nan_to_num(df["close"].values)
        atr = np.nan_to_num(indicators["atr"])
        
        # Entry conditions
        entry_short = (close < lower_band) & (rsi < params["rsi_oversold"])
        
        # Exit condition
        exit_short = close > middle_band
        
        # Create signal array
        short_entry = np.zeros_like(rsi)
        short_entry[entry_short] = -1.0
        short_exit = np.zeros_like(rsi)
        short_exit[exit_short] = 0.0  # Flat signal
        
        # Combine entry and exit
        signals.iloc[:] = 0.0
        signals.iloc[entry_short] = -1.0
        signals.iloc[exit_short] = 0.0
        
        # Ensure no conflicting signals
        for i in range(1, len(signals)):
            if signals.iloc[i] == -1.0:
                # Check if we already have a short signal
                if signals.iloc[i-1] == -1.0:
                    signals.iloc[i] = 0.0  # Prevent re-entry
            elif signals.iloc[i] == 0.0:
                # If we were in a short position and price crosses middle band
                if signals.iloc[i-1] == -1.0:
                    signals.iloc[i] = 0.0  # Flat signal
        
        # Ensure only one signal per bar
        for i in range(len(signals) - 1, 0, -1):
            if signals.iloc[i] == 0.0 and signals.iloc[i-1] == -1.0:
                signals.iloc[i] = 0.0
            elif signals.iloc[i] == -1.0 and signals.iloc[i-1] == -1.0:
                signals.iloc[i] = 0.0
                
        return signals