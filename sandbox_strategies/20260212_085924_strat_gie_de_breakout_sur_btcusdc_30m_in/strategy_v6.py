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
        return ["bollinger", "supertrend", "atr", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(20, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.5),
            "tp_atr_mult": ParameterSpec(3.0, 10.0, 0.5),
            "warmup": ParameterSpec(20, 100, 5),
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
        st = indicators["supertrend"]
        atr = indicators["atr"]
        rsi = indicators["rsi"]
        
        # Extract BB components
        bb_upper = bb["upper"]
        bb_lower = bb["lower"]
        bb_middle = bb["middle"]
        
        # Extract Supertrend components
        st_direction = st["direction"]
        st_trend = st["supertrend"]
        
        # Previous values
        prev_close = df["close"].shift(1).values
        prev_bb_upper = np.roll(bb_upper, 1)
        prev_bb_lower = np.roll(bb_lower, 1)
        prev_st_direction = np.roll(st_direction, 1)
        prev_rsi = np.roll(rsi, 1)
        
        # Volume
        volume = df["volume"].values
        volume_sma = pd.Series(volume).rolling(20).mean().values
        
        # Entry conditions
        entry_long = (
            (df["close"].values > bb_upper) &
            (prev_close <= prev_bb_upper) &
            (st_direction == 1) &
            (rsi > params["rsi_oversold"]) &
            (volume > volume_sma)
        )
        
        # Exit conditions
        exit_long = (
            (df["close"].values < bb_lower) |
            (st_direction == -1) |
            (rsi > params["rsi_overbought"])
        )
        
        # Generate signals
        long_entries = np.where(entry_long, 1.0, 0.0)
        long_exits = np.where(exit_long, 0.0, 1.0)
        
        # Combine entry and exit signals
        signals.iloc[:] = long_entries * long_exits
        
        return signals