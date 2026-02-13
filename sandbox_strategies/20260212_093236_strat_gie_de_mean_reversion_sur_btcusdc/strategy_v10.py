from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="keltner_cci_mean_reversion_v10")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "cci", "atr", "momentum"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"cci_period": 14, "keltner_multiplier": 1.5, "keltner_period": 20, "momentum_period": 10, "stop_atr_mult": 1.0, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "cci_period": ParameterSpec(5, 30, 1),
            "keltner_multiplier": ParameterSpec(0.5, 3.0, 0.1),
            "keltner_period": ParameterSpec(10, 50, 1),
            "momentum_period": ParameterSpec(5, 20, 1),
            "stop_atr_mult": ParameterSpec(0.5, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec(1.0, 4.0, 0.1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        keltner = indicators["keltner"]
        cci = np.nan_to_num(indicators["cci"])
        atr = np.nan_to_num(indicators["atr"])
        momentum = np.nan_to_num(indicators["momentum"])
        
        # Keltner channels
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_middle = np.nan_to_num(keltner["middle"])
        
        # Entry conditions
        price = np.nan_to_num(df["close"].values)
        entry_condition = (price >= keltner_upper) & (cci < 0) & (momentum < 0)
        
        # Exit condition
        exit_condition = price <= keltner_middle
        
        # Generate signals
        entry_indices = np.where(entry_condition)[0]
        exit_indices = np.where(exit_condition)[0]
        
        # Set long signals
        for i in entry_indices:
            if i < len(signals):
                signals.iloc[i] = 1.0
                
        # Set flat signals on exit
        for i in exit_indices:
            if i < len(signals) and signals.iloc[i] == 1.0:
                signals.iloc[i] = 0.0
                
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals