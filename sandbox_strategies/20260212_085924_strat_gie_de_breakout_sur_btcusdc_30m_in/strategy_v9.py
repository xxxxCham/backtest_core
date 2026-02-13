from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bolinger_supertrend_atr_breakout_revised")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "supertrend", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                param_type="float",
                min_value=1.0,
                max_value=5.0,
                step=0.5,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                param_type="float",
                min_value=3.0,
                max_value=10.0,
                step=0.5,
            ),
            "warmup": ParameterSpec(
                name="warmup",
                param_type="int",
                min_value=20,
                max_value=100,
                step=10,
            ),
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
        bb = indicators["bollinger"]
        bb_upper = np.nan_to_num(bb["upper"])
        bb_lower = np.nan_to_num(bb["lower"])
        bb_middle = np.nan_to_num(bb["middle"])
        
        st = indicators["supertrend"]
        st_direction = np.nan_to_num(st["direction"])
        st_trend = np.nan_to_num(st["supertrend"])
        
        atr = np.nan_to_num(indicators["atr"])
        
        # Previous values
        prev_bb_upper = np.roll(bb_upper, 1)
        prev_bb_lower = np.roll(bb_lower, 1)
        prev_close = np.roll(df["close"].values, 1)
        prev_st_direction = np.roll(st_direction, 1)
        
        # Entry condition: close breaks above upper band, previous close was below or equal
        # Supertrend confirms uptrend
        entry_long = (
            (df["close"].values > bb_upper) &
            (prev_close <= prev_bb_upper) &
            (st_direction == 1)
        )
        
        # Exit condition: close breaks below lower band or supertrend changes direction
        exit_long = (
            (df["close"].values < bb_lower) |
            (st_direction == -1)
        )
        
        # Generate signals
        entry_indices = np.where(entry_long)[0]
        exit_indices = np.where(exit_long)[0]
        
        for i in entry_indices:
            if i >= warmup:
                signals.iloc[i] = 1.0  # LONG signal
                
        # Set exit signals
        for i in exit_indices:
            if i >= warmup:
                if signals.iloc[i-1] == 1.0:
                    signals.iloc[i] = 0.0  # FLAT signal
                    
        return signals