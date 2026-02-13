from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="keltner_cci_mean_reversion_v6")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "cci", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"cci_period": 14, "keltner_multiplier": 1.5, "keltner_period": 20, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.0, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "cci_period": ParameterSpec(10, 30, 1),
            "keltner_multiplier": ParameterSpec(1.0, 3.0, 0.1),
            "keltner_period": ParameterSpec(10, 50, 1),
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
        keltner = indicators["keltner"]
        cci = np.nan_to_num(indicators["cci"])
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Keltner bands
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_middle = np.nan_to_num(keltner["middle"])
        keltner_lower = np.nan_to_num(keltner["lower"])
        
        # CCI logic
        cci_shifted = np.roll(cci, 1)
        cci_shifted[0] = 0.0
        
        # Entry conditions
        entry_long = (
            (df["close"] >= keltner_upper) &
            (cci < -100) &
            (cci > cci_shifted) &
            (cci_shifted < -50) &
            (rsi < params["rsi_oversold"])
        )
        
        # Exit condition
        exit_long = df["close"] <= keltner_middle
        
        # Generate signals
        long_entries = np.where(entry_long, 1.0, 0.0)
        long_exits = np.where(exit_long, 0.0, 1.0)
        
        # Combine signals
        signals = pd.Series(long_entries * long_exits, index=df.index, dtype=np.float64)
        
        return signals