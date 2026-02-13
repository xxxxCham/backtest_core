from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bolinger_supertrend_atr_breakout")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "supertrend", "atr", "volume_oscillator"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "stop_atr_mult": ParameterSpec("stop_atr_mult", 1.0, 5.0, 0.5),
            "tp_atr_mult": ParameterSpec("tp_atr_mult", 3.0, 10.0, 0.5),
            "warmup": ParameterSpec("warmup", 30, 100, 10),
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
        st = indicators["supertrend"]
        atr = np.nan_to_num(indicators["atr"])
        vol_osc = np.nan_to_num(indicators["volume_oscillator"])
        
        # Extract Bollinger bands
        bb_upper = np.nan_to_num(bb["upper"])
        bb_lower = np.nan_to_num(bb["lower"])
        bb_middle = np.nan_to_num(bb["middle"])
        
        # Extract Supertrend
        st_direction = np.nan_to_num(st["direction"])
        st_trend = np.nan_to_num(st["supertrend"])
        
        # Previous values
        prev_bb_upper = np.roll(bb_upper, 1)
        prev_bb_lower = np.roll(bb_lower, 1)
        prev_close = np.roll(df["close"], 1)
        prev_st_direction = np.roll(st_direction, 1)
        
        # Entry condition: breakout above upper band, previous close was below, supertrend direction up, volume positive
        entry_long = (df["close"].values > bb_upper) & (prev_close <= prev_bb_upper) & (st_direction == 1) & (vol_osc > 0)
        
        # Exit condition: close below lower band OR supertrend direction down OR bollinger contraction
        bb_contracted = (bb_upper - bb_lower) < np.roll((bb_upper - bb_lower), 1)
        exit_long = (df["close"].values < bb_lower) | (st_direction == -1) | (bb_contracted)
        
        # Generate signals
        entry_indices = np.where(entry_long)[0]
        exit_indices = np.where(exit_long)[0]
        
        for i in entry_indices:
            if i > 0:
                signals.iloc[i] = 1.0
                
        # Set exit signals
        for i in exit_indices:
            if i > 0:
                signals.iloc[i] = 0.0
                
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals