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
        return ["bollinger", "supertrend", "atr", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec("rsi_overbought", 50, 90, 1),
            "rsi_oversold": ParameterSpec("rsi_oversold", 10, 40, 1),
            "rsi_period": ParameterSpec("rsi_period", 5, 30, 1),
            "stop_atr_mult": ParameterSpec("stop_atr_mult", 1.0, 5.0, 0.5),
            "tp_atr_mult": ParameterSpec("tp_atr_mult", 3.0, 10.0, 0.5),
            "warmup": ParameterSpec("warmup", 20, 100, 10),
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
        atr = np.nan_to_num(indicators["atr"])
        rsi = np.nan_to_num(indicators["rsi"])
        
        # Bollinger bands
        bb_upper = np.nan_to_num(bb["upper"])
        bb_lower = np.nan_to_num(bb["lower"])
        bb_middle = np.nan_to_num(bb["middle"])
        
        # Supertrend
        st_direction = np.nan_to_num(st["direction"])
        
        # Calculate band width
        bb_width = bb_upper - bb_lower
        bb_width_prev = np.roll(bb_width, 1)
        
        # Close prices
        close = np.nan_to_num(df["close"].values)
        prev_close = np.roll(close, 1)
        prev_bb_upper = np.roll(bb_upper, 1)
        
        # Entry condition: breakout above upper band, previous close below upper band, supertrend direction up, RSI oversold, band contraction
        entry_long = (close > bb_upper) & (prev_close <= prev_bb_upper) & (st_direction == 1) & (rsi < params["rsi_oversold"]) & (bb_width < (bb_width_prev * 0.9))
        
        # Exit condition: close below lower band, supertrend direction down, band expansion
        exit_long = (close < bb_lower) | (st_direction == -1) | (bb_width > (bb_width_prev * 1.2))
        
        # Generate signals
        entry_indices = np.where(entry_long)[0]
        exit_indices = np.where(exit_long)[0]
        
        # Mark entry points
        for idx in entry_indices:
            if idx >= warmup:
                signals.iloc[idx] = 1.0
        
        # Mark exit points
        for idx in exit_indices:
            if idx >= warmup:
                signals.iloc[idx] = 0.0
                
        return signals