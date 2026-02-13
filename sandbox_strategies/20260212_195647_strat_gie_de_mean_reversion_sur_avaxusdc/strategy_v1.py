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
        return ["bollinger", "stoch_rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(70, 80, 1),
            "rsi_oversold": ParameterSpec(20, 30, 1),
            "rsi_period": ParameterSpec(10, 20, 1),
            "stop_atr_mult": ParameterSpec(1.0, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 4.0, 0.1),
            "warmup": ParameterSpec(30, 70, 1)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        bb = indicators["bollinger"]
        stoch_rsi = indicators["stoch_rsi"]
        atr = np.nan_to_num(indicators["atr"])
        
        lower_bb = np.nan_to_num(bb["lower"])
        middle_bb = np.nan_to_num(bb["middle"])
        price = np.nan_to_num(df["close"].values)
        
        stoch_rsi_k = np.nan_to_num(stoch_rsi["k"])
        stoch_rsi_d = np.nan_to_num(stoch_rsi["d"])
        
        # Entry conditions
        entry_condition = (price < lower_bb) & (stoch_rsi_k < 20) & (stoch_rsi_d < 20)
        
        # Exit condition
        exit_condition = price > middle_bb
        
        # Generate signals
        entry_indices = np.where(entry_condition)[0]
        exit_indices = np.where(exit_condition)[0]
        
        for i in entry_indices:
            signals.iloc[i] = 1.0
        
        for i in exit_indices:
            if signals.iloc[i] == 1.0:
                signals.iloc[i] = 0.0
                
        return signals