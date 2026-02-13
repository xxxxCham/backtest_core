from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="supertrend_adx_atr_trend_following")

    @property
    def required_indicators(self) -> List[str]:
        return ["supertrend", "adx", "atr", "ema"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"adx_threshold": 20, "atr_threshold": 100, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_threshold": ParameterSpec(param_type="int", min_value=10, max_value=30, step=5),
            "atr_threshold": ParameterSpec(param_type="float", min_value=50.0, max_value=200.0, step=25.0),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=2.0, step=0.25),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=4.0, step=0.5),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        supertrend_direction = np.nan_to_num(indicators["supertrend"]["direction"])
        adx_value = np.nan_to_num(indicators["adx"]["adx"])
        atr_value = np.nan_to_num(indicators["atr"])
        ema_value = np.nan_to_num(indicators["ema"])
        close = np.nan_to_num(df["close"].values)
        
        # Extract params
        adx_threshold = params.get("adx_threshold", 20)
        atr_threshold = params.get("atr_threshold", 100)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))
        
        # Entry condition: supertrend direction positive, ADX above threshold, price above EMA, ATR above threshold
        entry_condition = (
            (supertrend_direction > 0) &
            (adx_value > adx_threshold) &
            (close > ema_value) &
            (atr_value > atr_threshold)
        )
        
        # Exit condition: supertrend direction negative
        exit_condition = (supertrend_direction < 0)
        
        # Generate signals
        long_entries = np.where(entry_condition, 1.0, 0.0)
        long_exits = np.where(exit_condition, 0.0, 0.0)
        
        # Combine signals
        signals = pd.Series(long_entries, index=df.index, dtype=np.float64)
        
        # Apply exit signals
        for i in range(1, len(signals)):
            if signals.iloc[i-1] == 1.0 and long_exits[i] == 0.0:
                signals.iloc[i] = 1.0
            elif signals.iloc[i-1] == 1.0 and long_exits[i] == 0.0:
                signals.iloc[i] = 1.0
            elif signals.iloc[i-1] == 1.0 and long_exits[i] == 0.0:
                signals.iloc[i] = 1.0
            else:
                signals.iloc[i] = 0.0
                
        # Warmup protection
        signals.iloc[:warmup] = 0.0
        
        return signals