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
        return {"cci_period": 14, "cci_threshold": 100, "keltner_multiplier": 1.5, "keltner_period": 20, "stop_atr_mult": 1.0, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "cci_period": ParameterSpec("cci_period", 10, 30, 1),
            "cci_threshold": ParameterSpec("cci_threshold", 50, 200, 10),
            "keltner_multiplier": ParameterSpec("keltner_multiplier", 1.0, 3.0, 0.1),
            "keltner_period": ParameterSpec("keltner_period", 10, 50, 5),
            "stop_atr_mult": ParameterSpec("stop_atr_mult", 0.5, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec("tp_atr_mult", 1.0, 4.0, 0.1),
            "warmup": ParameterSpec("warmup", 20, 100, 10),
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
        
        close = np.nan_to_num(df["close"].values)
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_lower = np.nan_to_num(keltner["lower"])
        keltner_middle = np.nan_to_num(keltner["middle"])
        
        # Volatility filter
        atr_mean = np.nan_to_num(pd.Series(atr).rolling(20).mean().shift(1).values)
        volatility_filter = atr > atr_mean
        
        # Entry conditions
        long_entry = (
            (close < keltner_lower) & 
            (cci < -cci_threshold) & 
            volatility_filter
        ) | (
            (close > keltner_upper) & 
            (cci > cci_threshold) & 
            volatility_filter
        )
        
        # Exit condition
        exit_condition = (close > keltner_middle) & (np.roll(close, 1) < keltner_middle)
        
        # Generate signals
        entry_indices = np.where(long_entry)[0]
        exit_indices = np.where(exit_condition)[0]
        
        for i in entry_indices:
            if i > 0:
                signals.iloc[i] = 1.0  # LONG signal
                
        # Set exit signals
        for i in exit_indices:
            if i > 0:
                signals.iloc[i] = 0.0  # FLAT signal
                
        # Warmup protection
        signals.iloc[:warmup] = 0.0
        
        return signals