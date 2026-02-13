from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_supertrend_atr_breakout_with_contract_filter")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "supertrend", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 90, 1),
            "rsi_oversold": ParameterSpec(10, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
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
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Extract indicators
        bb = indicators["bollinger"]
        upper_band = np.nan_to_num(bb["upper"])
        lower_band = np.nan_to_num(bb["lower"])
        middle_band = np.nan_to_num(bb["middle"])
        
        st = indicators["supertrend"]
        supertrend_direction = np.nan_to_num(st["direction"])
        supertrend_value = np.nan_to_num(st["supertrend"])
        
        atr = np.nan_to_num(indicators["atr"])
        
        close = np.nan_to_num(df["close"].values)
        volume = np.nan_to_num(df["volume"].values)
        
        # Bollinger Band contracting filter (using ratio of current width to 20-period average)
        bb_width = upper_band - lower_band
        bb_width_avg = np.convolve(bb_width, np.ones(20)/20, mode='valid')
        bb_width_avg = np.pad(bb_width_avg, (len(bb_width) - len(bb_width_avg), 0), mode='edge')
        bb_contracting = (bb_width / (bb_width_avg + 1e-8)) < 1.2
        
        # Entry conditions
        long_condition = (close > upper_band) & (supertrend_direction > 0) & (bb_contracting)
        short_condition = (close < lower_band) & (supertrend_direction < 0) & (bb_contracting)
        
        # Generate signals
        long_signals = pd.Series(0.0, index=df.index)
        short_signals = pd.Series(0.0, index=df.index)
        
        long_signals[long_condition] = 1.0
        short_signals[short_condition] = -1.0
        
        # Combine signals
        signals = long_signals + short_signals
        
        return signals