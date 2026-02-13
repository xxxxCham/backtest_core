from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="keltner_cci_mean_reversion_v4")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "cci", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"cci_period": 14, "keltner_multiplier": 1.5, "keltner_period": 20, "stop_atr_mult": 1.0, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "cci_period": ParameterSpec(param_name="cci_period", param_type="int", min_value=5, max_value=30, step=1),
            "keltner_multiplier": ParameterSpec(param_name="keltner_multiplier", param_type="float", min_value=1.0, max_value=3.0, step=0.1),
            "keltner_period": ParameterSpec(param_name="keltner_period", param_type="int", min_value=10, max_value=50, step=5),
            "stop_atr_mult": ParameterSpec(param_name="stop_atr_mult", param_type="float", min_value=0.5, max_value=2.0, step=0.1),
            "tp_atr_mult": ParameterSpec(param_name="tp_atr_mult", param_type="float", min_value=1.0, max_value=4.0, step=0.1),
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
        
        # Keltner bands
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_middle = np.nan_to_num(keltner["middle"])
        keltner_lower = np.nan_to_num(keltner["lower"])
        
        # CCI shifted for turning point detection
        cci_shifted = np.roll(cci, 1)
        cci_shifted[0] = cci[0]
        
        # Entry conditions
        price = np.nan_to_num(df["close"].values)
        entry_long_condition = (price > keltner_upper) & (cci < -100) & (cci_shifted > cci) & (cci_shifted > -50)
        
        # Exit condition
        exit_condition = price < keltner_middle
        
        # Signal generation
        entry_indices = np.where(entry_long_condition)[0]
        for idx in entry_indices:
            signals.iloc[idx] = 1.0  # LONG signal
            
        # Set warmup period
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals