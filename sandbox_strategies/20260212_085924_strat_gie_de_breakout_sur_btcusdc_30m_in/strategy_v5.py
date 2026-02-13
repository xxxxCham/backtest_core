from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bolinger_supertrend_atr_breakout_improved")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "supertrend", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=5.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=3.0, max_value=10.0, step=0.5),
            "warmup": ParameterSpec(param_type="int", min_value=20, max_value=100, step=10)
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
        st = indicators["supertrend"]
        atr = np.nan_to_num(indicators["atr"])
        
        close = np.nan_to_num(df["close"].values)
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        bb_upper = np.nan_to_num(bb["upper"])
        prev_bb_upper = np.roll(bb_upper, 1)
        prev_bb_upper[0] = bb_upper[0]
        
        st_direction = np.nan_to_num(st["direction"])
        prev_st_direction = np.roll(st_direction, 1)
        prev_st_direction[0] = st_direction[0]
        
        # Entry condition: breakout above upper band, previous close was below upper band, supertrend direction is up
        entry_condition = (close > bb_upper) & (prev_close <= prev_bb_upper) & (st_direction == 1)
        
        # Exit condition: close below lower band or supertrend direction changes to down
        bb_lower = np.nan_to_num(bb["lower"])
        exit_condition = (close < bb_lower) | (st_direction == -1)
        
        # Generate signals
        entry_points = np.where(entry_condition, 1.0, 0.0)
        exit_points = np.where(exit_condition, -1.0, 0.0)
        
        # Combine signals
        signals = pd.Series(entry_points, index=df.index, dtype=np.float64)
        signals = signals.where(~exit_points.astype(bool), -1.0)
        
        return signals