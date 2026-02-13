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
        return ["bollinger", "supertrend", "atr", "volume_oscillator"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 6.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1),
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
        
        # Get arrays
        close = np.nan_to_num(df["close"].values)
        open_ = np.nan_to_num(df["open"].values)
        volume = np.nan_to_num(df["volume"].values)
        
        # Bollinger bands
        bb_upper = np.nan_to_num(bb["upper"])
        bb_lower = np.nan_to_num(bb["lower"])
        bb_middle = np.nan_to_num(bb["middle"])
        
        # Supertrend
        st_direction = np.nan_to_num(st["direction"])
        st_trend = np.nan_to_num(st["supertrend"])
        
        # Volume and SMA
        volume_sma = np.nan_to_num(pd.Series(volume).rolling(window=20).mean().values)
        
        # Previous values
        prev_close = np.roll(close, 1)
        prev_bb_upper = np.roll(bb_upper, 1)
        prev_bb_lower = np.roll(bb_lower, 1)
        prev_st_direction = np.roll(st_direction, 1)
        prev_vol_osc = np.roll(vol_osc, 1)
        
        # Entry conditions
        entry_long = (
            (close > bb_upper) &
            (prev_close <= prev_bb_upper) &
            (st_direction == 1) &
            (volume > volume_sma) &
            (vol_osc > 0)
        )
        
        # Exit conditions
        exit_long = (
            (close < bb_lower) |
            (st_direction == -1) |
            (vol_osc < 0)
        )
        
        # Initialize entry and exit masks
        entry_mask = np.zeros_like(entry_long, dtype=bool)
        exit_mask = np.zeros_like(exit_long, dtype=bool)
        
        # Apply entry conditions
        entry_mask[entry_long] = True
        
        # Apply exit conditions
        exit_mask[exit_long] = True
        
        # Generate signals
        signals.iloc[entry_mask] = 1.0
        signals.iloc[exit_mask] = 0.0
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals