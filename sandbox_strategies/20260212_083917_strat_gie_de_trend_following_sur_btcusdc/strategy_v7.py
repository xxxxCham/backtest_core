from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_30m_trend_following_revised")

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
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=6.0, step=0.5),
            "warmup": ParameterSpec(param_type="int", min_value=30, max_value=100, step=10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        price = np.nan_to_num(df["close"].values)
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        bb_upper = np.nan_to_num(bb["upper"])
        bb_middle = np.nan_to_num(bb["middle"])
        bb_lower = np.nan_to_num(bb["lower"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Entry conditions for short
        entry_short = (price < bb_lower) & (rsi < rsi_oversold) & (price < bb_middle)
        
        # Exit conditions
        exit_short = (price > bb_middle) | (rsi > rsi_overbought) | (price > bb_upper)
        
        # Generate signals
        entry_short_indices = np.where(entry_short)[0]
        exit_short_indices = np.where(exit_short)[0]
        
        for i in range(len(entry_short_indices)):
            entry_idx = entry_short_indices[i]
            # Set signal to -1 for short entry
            signals.iloc[entry_idx] = -1.0
            
            # Find exit after entry
            future_exit = np.where((exit_short_indices > entry_idx) & (exit_short_indices < len(df)))[0]
            if len(future_exit) > 0:
                exit_idx = exit_short_indices[future_exit[0]]
                signals.iloc[exit_idx] = 0.0
        
        return signals