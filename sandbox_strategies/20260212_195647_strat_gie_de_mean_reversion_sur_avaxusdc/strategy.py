from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mean_reversion_bollinger_stoch_rsi_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "stoch_rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(20, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        close = np.nan_to_num(df["close"].values)
        bb = indicators["bollinger"]
        bb_upper = np.nan_to_num(bb["upper"])
        bb_middle = np.nan_to_num(bb["middle"])
        bb_lower = np.nan_to_num(bb["lower"])
        stoch_rsi = indicators["stoch_rsi"]
        stoch_rsi_k = np.nan_to_num(stoch_rsi["k"])
        stoch_rsi_d = np.nan_to_num(stoch_rsi["d"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Entry long conditions
        entry_long = (close < bb_lower) & (stoch_rsi_k < 20) & (stoch_rsi_k > stoch_rsi_d) & (close < bb_middle)
        
        # Exit condition
        exit_long = close > bb_middle
        
        # Generate signals
        entry_indices = np.where(entry_long)[0]
        exit_indices = np.where(exit_long)[0]
        
        for i in range(len(entry_indices)):
            entry_idx = entry_indices[i]
            signals.iloc[entry_idx] = 1.0
            
            # Find next exit
            next_exit = np.where((exit_indices > entry_idx) & (exit_indices < len(signals)))[0]
            if len(next_exit) > 0:
                exit_idx = exit_indices[next_exit[0]]
                signals.iloc[exit_idx] = 0.0
                
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        return signals