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
        return {"cci_period": 14, "cci_threshold": 100, "keltner_multiplier": 1.5, "keltner_period": 20, "stop_atr_mult": 1.0, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "cci_period": ParameterSpec(5, 30, 1),
            "cci_threshold": ParameterSpec(50, 200, 10),
            "keltner_multiplier": ParameterSpec(1.0, 3.0, 0.1),
            "keltner_period": ParameterSpec(10, 50, 5),
            "stop_atr_mult": ParameterSpec(0.5, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec(1.0, 5.0, 0.5),
            "warmup": ParameterSpec(20, 100, 10)
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
        open_ = np.nan_to_num(df["open"].values)
        
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_middle = np.nan_to_num(keltner["middle"])
        keltner_lower = np.nan_to_num(keltner["lower"])
        
        # Shifted CCI for crossover condition
        cci_prev = np.roll(cci, 1)
        cci_prev[0] = 0
        
        # Entry conditions
        entry_long = (close < keltner_lower) & (cci < -cci_threshold) & (cci_prev > -cci_threshold) & (close > open_) & (close > keltner_middle)
        
        # Exit condition
        exit_long = close > keltner_middle
        
        # Initialize entry/exit signals
        entry_signal = pd.Series(0.0, index=df.index)
        exit_signal = pd.Series(0.0, index=df.index)
        
        # Set entry signals
        entry_signal[entry_long] = 1.0
        
        # Set exit signals
        exit_signal[exit_long] = -1.0
        
        # Combine signals
        signals = entry_signal + exit_signal
        
        # Ensure warmup period
        signals.iloc[:warmup] = 0.0
        
        return signals