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
            "rsi_overbought": ParameterSpec(param_type="int", min_value=60, max_value=80, step=1),
            "rsi_oversold": ParameterSpec(param_type="int", min_value=10, max_value=30, step=1),
            "rsi_period": ParameterSpec(param_type="int", min_value=10, max_value=20, step=1),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=2.0, step=0.1),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=4.0, step=0.1),
            "warmup": ParameterSpec(param_type="int", min_value=30, max_value=100, step=10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        rsi = np.nan_to_num(indicators["rsi"])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_middle = np.nan_to_num(indicators["bollinger"]["middle"])
        atr = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)
        
        rsi_overbought = params.get("rsi_overbought", 70)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))
        
        # Entry condition: price touches upper bollinger band, RSI in overbought, RSI trending down, ATR increasing
        entry_condition = (
            (close >= bb_upper * 0.99) &
            (rsi > rsi_overbought) &
            (rsi < np.roll(rsi, 1)) &
            (atr > np.roll(atr, 1)) &
            (rsi < np.roll(rsi, 2))
        )
        
        # Exit when price returns to middle bollinger band
        exit_condition = close <= bb_middle
        
        # Generate long signals
        long_entries = entry_condition
        long_exits = exit_condition
        
        # Initialize signal array
        signal_array = np.zeros_like(close)
        
        # Set long entries
        signal_array[long_entries] = 1.0
        
        # Set exits
        for i in range(len(signal_array)):
            if signal_array[i] == 1.0:
                # Look for exit condition after entry
                for j in range(i + 1, len(signal_array)):
                    if long_exits[j]:
                        signal_array[j] = 0.0
                        break
        
        signals = pd.Series(signal_array, index=df.index, dtype=np.float64)
        signals.iloc[:warmup] = 0.0
        return signals