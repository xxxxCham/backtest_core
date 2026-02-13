from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="keltner_cci_mean_reversion_v5")

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
        signals.iloc[:warmup] = 0.0
        
        keltner = indicators["keltner"]
        cci = np.nan_to_num(indicators["cci"])
        atr = np.nan_to_num(indicators["atr"])
        
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_middle = np.nan_to_num(keltner["middle"])
        keltner_lower = np.nan_to_num(keltner["lower"])
        
        price = np.nan_to_num(df["close"].values)
        
        # CCI lagged for divergence
        cci_shifted = np.roll(cci, 1)
        cci_shifted[0] = cci[0]
        
        # Entry condition: price touches upper Keltner band and CCI shows bearish divergence
        entry_long = (price >= keltner_upper) & (cci < -100) & (cci > cci_shifted) & (cci_shifted < -50)
        
        # Exit condition: price returns to Keltner middle
        exit_signal = price <= keltner_middle
        
        # Set signals
        long_entries = np.where(entry_long, 1.0, 0.0)
        signals = pd.Series(long_entries, index=df.index, dtype=np.float64)
        
        # Apply exit logic (simple flat when price hits middle)
        for i in range(1, len(signals)):
            if signals.iloc[i-1] == 1.0 and exit_signal[i]:
                signals.iloc[i] = 0.0
                
        return signals