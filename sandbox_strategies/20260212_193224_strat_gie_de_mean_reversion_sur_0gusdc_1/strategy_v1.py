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
        return {"cci_period": 14, "cci_threshold": 100, "keltner_multiplier": 1.5, "keltner_period": 20, "stop_atr_mult": 1.0, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "cci_period": ParameterSpec(5, 30, 1),
            "cci_threshold": ParameterSpec(50, 200, 10),
            "keltner_multiplier": ParameterSpec(1.0, 3.0, 0.1),
            "keltner_period": ParameterSpec(10, 50, 5),
            "stop_atr_mult": ParameterSpec(0.5, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec(1.0, 4.0, 0.1),
            "warmup": ParameterSpec(20, 100, 5)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        cci_period = int(params.get("cci_period", 14))
        cci_threshold = float(params.get("cci_threshold", 100))
        keltner_multiplier = float(params.get("keltner_multiplier", 1.5))
        keltner_period = int(params.get("keltner_period", 20))
        stop_atr_mult = float(params.get("stop_atr_mult", 1.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.0))
        warmup = int(params.get("warmup", 50))
        
        keltner = indicators["keltner"]
        cci = np.nan_to_num(indicators["cci"])
        atr = np.nan_to_num(indicators["atr"])
        
        upper_band = np.nan_to_num(keltner["upper"])
        middle_band = np.nan_to_num(keltner["middle"])
        lower_band = np.nan_to_num(keltner["lower"])
        
        # Previous values for cci cross detection
        cci_prev = np.roll(cci, 1)
        cci_prev = np.where(np.arange(len(cci_prev)) == 0, cci[0], cci_prev)
        
        # Entry conditions
        # Close below lower band and CCI crosses below -100
        entry_long_cond1 = (df["close"] < lower_band) & (cci_prev < -cci_threshold) & (cci >= -cci_threshold)
        
        # Close above upper band and CCI crosses above 100
        entry_long_cond2 = (df["close"] > upper_band) & (cci_prev > cci_threshold) & (cci <= cci_threshold)
        
        entry_long = entry_long_cond1 | entry_long_cond2
        
        # Exit when price crosses back into middle band
        exit_cond = (df["close"].shift(1) < middle_band) & (df["close"] > middle_band)
        
        # Generate signals
        entry_indices = np.where(entry_long)[0]
        exit_indices = np.where(exit_cond)[0]
        
        for i in entry_indices:
            signals.iloc[i] = 1.0  # LONG signal
            
        for i in exit_indices:
            if signals.iloc[i] == 1.0:
                signals.iloc[i] = 0.0  # FLAT signal
                
        # Set warmup period
        signals.iloc[:warmup] = 0.0
        
        return signals