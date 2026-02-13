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
        return ["keltner", "cci", "atr", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"cci_period": 14, "keltner_multiplier": 1.5, "keltner_period": 20, "rsi_period": 14, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "cci_period": ParameterSpec(10, 30, 1),
            "keltner_multiplier": ParameterSpec(1.0, 3.0, 0.1),
            "keltner_period": ParameterSpec(10, 50, 1),
            "rsi_period": ParameterSpec(10, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.1),
            "tp_atr_mult": ParameterSpec(3.0, 10.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1),
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
        rsi = np.nan_to_num(indicators["rsi"])
        
        # Keltner bands
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_middle = np.nan_to_num(keltner["middle"])
        keltner_lower = np.nan_to_num(keltner["lower"])
        
        # Calculate slope of middle band
        keltner_middle_lag1 = np.roll(keltner_middle, 1)
        keltner_middle_slope = keltner_middle - keltner_middle_lag1
        
        # Lagged CCI and RSI
        cci_lag1 = np.roll(cci, 1)
        rsi_lag1 = np.roll(rsi, 1)
        
        # Entry conditions
        entry_condition = (
            (df["close"] <= keltner_lower) &
            (cci < -100) &
            (cci_lag1 > -100) &
            (keltner_middle_slope < 0)
        )
        
        # Exit conditions
        exit_condition = (
            (df["close"] >= keltner_middle) &
            (rsi < 30) &
            (rsi_lag1 > 30)
        )
        
        # Generate signals
        long_entries = entry_condition
        long_exits = exit_condition
        
        # Signal logic
        entry_indices = np.where(long_entries)[0]
        exit_indices = np.where(long_exits)[0]
        
        for i in range(len(entry_indices)):
            entry_idx = entry_indices[i]
            signals.iloc[entry_idx] = 1.0
            
            # Find corresponding exit
            future_exits = exit_indices[exit_indices > entry_idx]
            if len(future_exits) > 0:
                exit_idx = future_exits[0]
                signals.iloc[exit_idx] = 0.0
                
        return signals