from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="keltner_cci_mean_reversion_v8")

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
            "tp_atr_mult": ParameterSpec(2.0, 10.0, 0.1),
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
        stop_atr_mult = float(params.get("stop_atr_mult", 2.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 5.0))
        warmup = int(params.get("warmup", 50))
        
        # Extract indicators
        keltner = indicators["keltner"]
        cci = np.nan_to_num(indicators["cci"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Keltner channel components
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_middle = np.nan_to_num(keltner["middle"])
        keltner_lower = np.nan_to_num(keltner["lower"])
        
        # Calculate slope of middle band
        keltner_middle_slope = np.gradient(keltner_middle)
        
        # CCI lagged for breakout confirmation
        cci_lag1 = np.roll(cci, 1)
        
        # Entry conditions for long
        entry_long = (df['close'] <= keltner_lower) & (cci < -100) & (cci_lag1 > -100) & (keltner_middle_slope < 0)
        
        # Exit condition when price returns to middle band
        exit_long = df['close'] >= keltner_middle
        
        # Initialize entry and exit signals
        entry_signal = pd.Series(0.0, index=df.index)
        exit_signal = pd.Series(0.0, index=df.index)
        
        # Mark entry signals
        entry_signal[entry_long] = 1.0
        
        # Mark exit signals
        exit_signal[exit_long] = -1.0
        
        # Combine entry and exit signals
        signals = entry_signal + exit_signal
        
        # Set warmup period to flat
        signals.iloc[:warmup] = 0.0
        
        # Ensure only long positions
        signals = signals.where(signals == 1.0, 0.0)
        
        return signals