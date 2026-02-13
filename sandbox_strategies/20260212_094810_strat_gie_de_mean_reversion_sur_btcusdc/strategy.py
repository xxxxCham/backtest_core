from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mean_reversion_bollinger_rsi_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(param_type="int", min_value=50, max_value=100, step=5),
            "rsi_oversold": ParameterSpec(param_type="int", min_value=0, max_value=50, step=5),
            "rsi_period": ParameterSpec(param_type="int", min_value=5, max_value=30, step=5),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=3.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=5.0, step=0.5),
            "warmup": ParameterSpec(param_type="int", min_value=20, max_value=100, step=10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        bb_upper = np.nan_to_num(bb["upper"])
        bb_middle = np.nan_to_num(bb["middle"])
        bb_lower = np.nan_to_num(bb["lower"])
        atr = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)
        
        # Get parameters
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_overbought = params.get("rsi_overbought", 70)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))
        
        # Entry condition: price touches lower bollinger band with oversold RSI and rising RSI
        entry_condition = (
            (close <= bb_lower) &
            (rsi < rsi_oversold) &
            (rsi > np.roll(rsi, 1)) &
            (rsi > np.roll(rsi, 2))
        )
        
        # Exit condition: price returns to middle bollinger band
        exit_condition = close >= bb_middle
        
        # Generate signals
        entry_indices = np.where(entry_condition)[0]
        exit_indices = np.where(exit_condition)[0]
        
        # Initialize signals to 0.0
        signals.iloc[:warmup] = 0.0
        
        # Set long signals
        for i in entry_indices:
            if i >= warmup:
                signals.iloc[i] = 1.0
        
        # Set flat signals at exit
        for i in exit_indices:
            if i >= warmup:
                signals.iloc[i] = 0.0
        
        return signals