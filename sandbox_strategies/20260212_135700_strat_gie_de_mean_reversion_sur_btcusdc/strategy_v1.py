from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="keltner_cci_mean_reversion")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "cci", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"atr_period": 14, "cci_period": 14, "keltner_period": 20, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "atr_period": ParameterSpec(5, 30, 1),
            "cci_period": ParameterSpec(5, 30, 1),
            "keltner_period": ParameterSpec(10, 40, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.5),
            "tp_atr_mult": ParameterSpec(3.0, 10.0, 0.5),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        keltner = indicators["keltner"]
        cci = indicators["cci"]
        atr = indicators["atr"]
        
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_middle = np.nan_to_num(keltner["middle"])
        keltner_lower = np.nan_to_num(keltner["lower"])
        cci_values = np.nan_to_num(cci)
        atr_values = np.nan_to_num(atr)
        
        # Entry condition: price below lower Keltner band, CCI shows reversal
        entry_condition = (df["close"] < keltner_lower) & (cci_values < -100) & (cci_values > np.roll(cci_values, 1))
        
        # Exit condition: price crosses above middle Keltner band
        exit_condition = df["close"] > keltner_middle
        
        # Generate long signals
        long_entries = entry_condition
        long_exits = exit_condition
        
        # Apply signals
        signals[long_entries] = 1.0
        signals[long_exits] = 0.0
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals