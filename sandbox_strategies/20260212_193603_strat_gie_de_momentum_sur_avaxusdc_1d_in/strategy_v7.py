from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="momentum_avaxusdc_1d")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "macd", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(param_type="int", min_value=60, max_value=80, step=5),
            "rsi_oversold": ParameterSpec(param_type="int", min_value=10, max_value=30, step=5),
            "rsi_period": ParameterSpec(param_type="int", min_value=10, max_value=20, step=5),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=2.0, step=0.25),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=4.0, step=0.5),
            "warmup": ParameterSpec(param_type="int", min_value=30, max_value=70, step=10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        rsi = np.nan_to_num(indicators["rsi"])
        macd = indicators["macd"]
        macd_hist = np.nan_to_num(macd["histogram"])
        atr = np.nan_to_num(indicators["atr"])
        
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        warmup = int(params.get("warmup", 50))
        
        signals.iloc[:warmup] = 0.0
        
        # Entry long conditions
        long_condition = (rsi > rsi_oversold) & (rsi > np.roll(rsi, 1)) & (macd_hist > 0) & (rsi < 50)
        
        # Entry short conditions
        short_condition = (rsi < rsi_overbought) & (rsi < np.roll(rsi, 1)) & (macd_hist < 0) & (rsi > 50)
        
        # Exit conditions
        exit_long_condition = (rsi > rsi_overbought) | (rsi < rsi_oversold) | (rsi > np.roll(rsi, 1)) | (rsi < np.roll(rsi, 1))
        exit_short_condition = (rsi > rsi_overbought) | (rsi < rsi_oversold) | (rsi > np.roll(rsi, 1)) | (rsi < np.roll(rsi, 1))
        
        # Generate signals
        long_entries = np.where(long_condition, 1.0, 0.0)
        short_entries = np.where(short_condition, -1.0, 0.0)
        
        # Simple logic to avoid overlapping entries
        signals.iloc[:] = 0.0
        for i in range(1, len(signals)):
            if long_entries[i] == 1.0:
                signals.iloc[i] = 1.0
            elif short_entries[i] == -1.0:
                signals.iloc[i] = -1.0
            elif signals.iloc[i-1] == 1.0 and exit_long_condition[i]:
                signals.iloc[i] = 0.0
            elif signals.iloc[i-1] == -1.0 and exit_short_condition[i]:
                signals.iloc[i] = 0.0
                
        return signals