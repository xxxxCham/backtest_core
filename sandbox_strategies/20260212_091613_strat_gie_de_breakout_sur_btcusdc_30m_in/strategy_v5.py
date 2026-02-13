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
        return ["bollinger", "supertrend", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"stop_atr_mult": 1.5, "tp_atr_mult": 3.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 6.0, 0.1),
            "warmup": ParameterSpec(20, 100, 5),
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
        
        close = np.nan_to_num(df["close"].values)
        volume = np.nan_to_num(df["volume"].values)
        avg_volume = np.nan_to_num(pd.Series(volume).rolling(window=20).mean().values)
        
        # Bollinger Bands
        upper_band = np.nan_to_num(bb["upper"])
        lower_band = np.nan_to_num(bb["lower"])
        middle_band = np.nan_to_num(bb["middle"])
        
        # Supertrend
        supertrend_direction = np.nan_to_num(st["direction"])
        
        # Bollinger Band contraction logic
        bb_width = upper_band - lower_band
        bb_width_mean = pd.Series(bb_width).rolling(window=20).mean().values
        bb_contracted = bb_width < (bb_width_mean * 0.8)
        
        # Entry conditions
        long_condition = (close > upper_band) & (supertrend_direction > 0) & (~bb_contracted) & (volume > avg_volume * 1.5)
        short_condition = (close < lower_band) & (supertrend_direction < 0) & (~bb_contracted) & (volume > avg_volume * 1.5)
        
        # Generate signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals