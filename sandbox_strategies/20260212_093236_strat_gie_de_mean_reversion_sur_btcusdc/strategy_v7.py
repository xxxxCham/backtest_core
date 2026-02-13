from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="keltner_cci_mean_reversion_v7")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "cci", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"cci_period": 14, "keltner_multiplier": 1.5, "keltner_period": 20, "stop_atr_mult": 1.0, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "cci_period": ParameterSpec("cci_period", 10, 30, 1),
            "keltner_multiplier": ParameterSpec("keltner_multiplier", 1.0, 3.0, 0.1),
            "keltner_period": ParameterSpec("keltner_period", 10, 50, 1),
            "stop_atr_mult": ParameterSpec("stop_atr_mult", 0.5, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec("tp_atr_mult", 1.0, 5.0, 0.1),
            "warmup": ParameterSpec("warmup", 20, 100, 5)
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
        cci = np.nan_to_num(indicators["cci"])
        atr = np.nan_to_num(indicators["atr"])
        
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_middle = np.nan_to_num(keltner["middle"])
        
        # Create shifted CCI for reversal detection
        cci_shifted = np.roll(cci, 1)
        cci_shifted[0] = cci[0]
        
        # Entry condition: price touches upper band and CCI shows bullish reversal
        entry_condition = (df['close'] >= keltner_upper) & (cci < -100) & (cci_shifted > -100) & (cci > cci_shifted)
        
        # Exit condition: price returns to middle band
        exit_condition = df['close'] <= keltner_middle
        
        # Generate signals
        entry_indices = np.where(entry_condition)[0]
        exit_indices = np.where(exit_condition)[0]
        
        # Mark long entries
        for idx in entry_indices:
            if idx >= warmup:
                signals.iloc[idx] = 1.0
        
        # Mark exits
        for idx in exit_indices:
            if idx >= warmup and signals.iloc[idx] == 1.0:
                signals.iloc[idx] = 0.0
        
        # Set warmup period
        signals.iloc[:warmup] = 0.0
        
        return signals