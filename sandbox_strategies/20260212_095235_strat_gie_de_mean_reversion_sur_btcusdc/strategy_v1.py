from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mean_reversion_keltner_cci")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "cci", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"cci_period": 14, "keltner_multiplier": 1.5, "keltner_period": 20, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "cci_period": ParameterSpec(10, 30, 1),
            "keltner_multiplier": ParameterSpec(1.0, 3.0, 0.1),
            "keltner_period": ParameterSpec(10, 50, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.1),
            "tp_atr_mult": ParameterSpec(3.0, 10.0, 0.1),
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
        stop_atr_mult = float(params.get("stop_atr_mult", 2.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 5.0))
        warmup = int(params.get("warmup", 50))
        
        keltner = indicators["keltner"]
        cci = np.nan_to_num(indicators["cci"])
        atr = np.nan_to_num(indicators["atr"])
        
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_middle = np.nan_to_num(keltner["middle"])
        
        price = np.nan_to_num(df["close"].values)
        
        # Entry condition: price at upper band, CCI shows reversal
        entry_condition = (
            (price > keltner_upper) &
            (cci > 100) &
            (np.roll(cci, 1) <= 100)
        )
        
        # Exit condition: price returns to middle band
        exit_condition = price < keltner_middle
        
        # Generate signals
        long_entries = entry_condition
        long_exits = exit_condition
        
        # Apply signals
        signals[long_entries] = 1.0
        signals[long_exits] = 0.0
        
        # Warmup protection
        signals.iloc[:warmup] = 0.0
        
        return signals