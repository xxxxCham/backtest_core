from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="keltner_cci_mean_reversion_v2")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "cci", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"cci_period": 14, "keltner_multiplier": 1.5, "keltner_period": 20, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "cci_period": ParameterSpec(5, 30, 1),
            "keltner_multiplier": ParameterSpec(1.0, 3.0, 0.1),
            "keltner_period": ParameterSpec(10, 50, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 10.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1)
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
        atr = np.nan_to_num(indicators["atr"])
        
        # Keltner bands
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_middle = np.nan_to_num(keltner["middle"])
        keltner_lower = np.nan_to_num(keltner["lower"])
        
        # CCI lagged
        cci_lag1 = np.roll(cci, 1)
        cci_lag2 = np.roll(cci, 2)
        
        # ATR mean
        atr_mean = np.mean(atr)
        
        # Entry condition: price below lower band, CCI crosses from negative to positive, and volatility filter
        entry_condition = (df["close"] < keltner_lower) & (cci < 0) & (cci_lag1 > 0) & (atr > atr_mean * 1.2)
        
        # Exit condition: price crosses above middle band with CCI turning positive
        exit_condition = (df["close"] > keltner_middle) & (cci > 0)
        
        # Set signals
        entry_indices = np.where(entry_condition)[0]
        for i in entry_indices:
            signals.iloc[i] = 1.0  # LONG
            
        # Exit when price returns to middle band with confirmation of weakening momentum
        exit_indices = np.where(exit_condition)[0]
        for i in exit_indices:
            if signals.iloc[i] == 1.0:
                signals.iloc[i] = 0.0  # FLAT
                
        return signals