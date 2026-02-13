from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="keltner_cci_mean_reversion_v3")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "cci", "atr", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"cci_period": 14, "cci_threshold": 100, "keltner_multiplier": 1.5, "keltner_period": 20, "rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "cci_period": ParameterSpec(5, 30, 1),
            "cci_threshold": ParameterSpec(50, 200, 10),
            "keltner_multiplier": ParameterSpec(1.0, 3.0, 0.1),
            "keltner_period": ParameterSpec(10, 50, 5),
            "rsi_overbought": ParameterSpec(60, 90, 5),
            "rsi_oversold": ParameterSpec(10, 40, 5),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
            "warmup": ParameterSpec(20, 100, 10)
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
        
        # Keltner channel boundaries
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_middle = np.nan_to_num(keltner["middle"])
        keltner_lower = np.nan_to_num(keltner["lower"])
        
        # Compute ATR mean for volatility filter
        atr_mean = np.mean(atr)
        
        # Entry conditions
        cci_long_condition = (cci < -params["cci_threshold"])
        rsi_long_condition = (rsi < params["rsi_oversold"])
        keltner_long_condition = (df["close"].values < keltner_lower)
        volatility_condition = (atr > atr_mean)
        
        # Combine all long entry conditions
        entry_long = (
            keltner_long_condition &
            cci_long_condition &
            rsi_long_condition &
            volatility_condition
        )
        
        # Exit condition
        close = np.nan_to_num(df["close"].values)
        exit_condition = (
            (close > keltner_middle) &
            (np.roll(close, 1) < keltner_middle)
        )
        
        # Generate signals
        entry_indices = np.where(entry_long)[0]
        exit_indices = np.where(exit_condition)[0]
        
        # Initialize signals array
        signals_values = signals.values
        
        # Set entry signals
        for i in entry_indices:
            if i > 0:
                signals_values[i] = 1.0  # LONG signal
        
        # Set exit signals
        for i in exit_indices:
            if i > 0 and signals_values[i-1] == 1.0:
                signals_values[i] = 0.0  # FLAT signal
                
        signals = pd.Series(signals_values, index=df.index, dtype=np.float64)
        return signals