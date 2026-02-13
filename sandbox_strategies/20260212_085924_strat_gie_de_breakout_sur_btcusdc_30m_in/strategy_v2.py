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
        return ["bollinger", "supertrend", "atr", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(param_type="int", min_value=50, max_value=90, step=5),
            "rsi_oversold": ParameterSpec(param_type="int", min_value=10, max_value=50, step=5),
            "rsi_period": ParameterSpec(param_type="int", min_value=5, max_value=30, step=5),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=5.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_typeparam_type="float", min_value=2.0, max_value=10.0, step=1.0),
            "warmup": ParameterSpec(param_type="int", min_value=20, max_value=100, step=10),
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
        supertrend = indicators["supertrend"]
        atr = np.nan_to_num(indicators["atr"])
        rsi = np.nan_to_num(indicators["rsi"])
        
        # Bollinger Bands
        bb_upper = np.nan_to_num(bb["upper"])
        bb_lower = np.nan_to_num(bb["lower"])
        bb_middle = np.nan_to_num(bb["middle"])
        
        # Supertrend
        st_direction = np.nan_to_num(supertrend["direction"])
        
        # Close and previous close
        close = np.nan_to_num(df["close"].values)
        prev_close = np.roll(close, 1)
        prev_bb_upper = np.roll(bb_upper, 1)
        prev_bb_lower = np.roll(bb_lower, 1)
        
        # Entry conditions
        entry_long_condition = (
            (close > bb_upper) &
            (prev_close <= prev_bb_upper) &
            (st_direction == 1) &
            (rsi < params["rsi_overbought"])
        )
        
        # Exit conditions
        exit_long_condition = (
            (close < bb_lower) |
            (close > bb_upper) |
            (st_direction == -1)
        )
        
        # Set signals
        entry_indices = np.where(entry_long_condition)[0]
        exit_indices = np.where(exit_long_condition)[0]
        
        # Assign long signals
        signals.iloc[entry_indices] = 1.0
        
        return signals