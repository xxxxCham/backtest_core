from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mean_reversion_bollinger_stoch_rsi_atri")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "stoch_rsi", "atr", "ema"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(20, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 4.0, 0.1),
            "warmup": ParameterSpec(20, 100, 5),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        bb = indicators["bollinger"]
        lower_bb = np.nan_to_num(bb["lower"])
        middle_bb = np.nan_to_num(bb["middle"])
        upper_bb = np.nan_to_num(bb["upper"])
        stoch_rsi_k = np.nan_to_num(indicators["stoch_rsi"]["k"])
        atr = np.nan_to_num(indicators["atr"])
        ema_50 = np.nan_to_num(indicators["ema"])
        close = np.nan_to_num(df["close"].values)
        
        # Entry conditions
        entry_long = (close < lower_bb) & (stoch_rsi_k < 20) & (close > ema_50)
        
        # Exit condition
        exit_long = close > middle_bb
        
        # Generate signals
        entry_indices = np.where(entry_long)[0]
        exit_indices = np.where(exit_long)[0]
        
        for i in range(len(entry_indices)):
            entry_idx = entry_indices[i]
            signals.iloc[entry_idx] = 1.0
            
            # Find next exit
            future_indices = exit_indices[exit_indices > entry_idx]
            if len(future_indices) > 0:
                exit_idx = future_indices[0]
                signals.iloc[exit_idx] = 0.0
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals