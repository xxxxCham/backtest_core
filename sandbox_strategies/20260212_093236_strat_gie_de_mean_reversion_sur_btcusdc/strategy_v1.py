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
        return {"cci_period": 14, "keltner_multiplier": 1.5, "keltner_period": 20, "stop_atr_mult": 1.0, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "cci_period": ParameterSpec(10, 30, 1),
            "keltner_multiplier": ParameterSpec(1.0, 3.0, 0.1),
            "keltner_period": ParameterSpec(10, 50, 1),
            "stop_atr_mult": ParameterSpec(0.5, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec(1.0, 4.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        cci_period = int(params.get("cci_period", 14))
        keltner_multiplier = float(params.get("keltner_multiplier", 1.5))
        keltner_period = int(params.get("keltner_period", 20))
        stop_atr_mult = float(params.get("stop_atr_mult", 1.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.0))
        warmup = int(params.get("warmup", 50))
        
        keltner = indicators["keltner"]
        cci = indicators["cci"]
        atr = indicators["atr"]
        
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_middle = np.nan_to_num(keltner["middle"])
        keltner_lower = np.nan_to_num(keltner["lower"])
        cci_values = np.nan_to_num(cci)
        atr_values = np.nan_to_num(atr)
        
        price = np.nan_to_num(df["close"].values)
        
        # Entry conditions
        price_crossed_above_upper = (price[:-1] < keltner_upper[:-1]) & (price[1:] >= keltner_upper[1:])
        cci_crossed_above_100 = (cci_values[:-1] < 100) & (cci_values[1:] >= 100)
        entry_long_condition = price_crossed_above_upper & cci_crossed_above_100
        
        # Exit condition
        price_crossed_middle = (price[:-1] > keltner_middle[:-1]) & (price[1:] <= keltner_middle[1:])
        exit_long_condition = price_crossed_middle
        
        # Generate signals
        entry_indices = np.where(entry_long_condition)[0]
        exit_indices = np.where(exit_long_condition)[0]
        
        for i in entry_indices:
            if i + 1 < len(signals):
                signals.iloc[i + 1] = 1.0  # LONG signal
                
        for i in exit_indices:
            if i + 1 < len(signals):
                signals.iloc[i + 1] = 0.0  # FLAT signal
                
        # Set warmup period to 0
        signals.iloc[:warmup] = 0.0
        
        return signals