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
        upper = np.nan_to_num(bb["upper"])
        lower = np.nan_to_num(bb["lower"])
        middle = np.nan_to_num(bb["middle"])
        
        st = indicators["supertrend"]
        supertrend = np.nan_to_num(st["supertrend"])
        direction = np.nan_to_num(st["direction"])
        
        atr = np.nan_to_num(indicators["atr"])
        
        # Volume
        volume = np.nan_to_num(df["volume"].values)
        avg_volume = np.convolve(volume, np.ones(10)/10, mode='valid')
        avg_volume = np.concatenate([np.full(9, np.nan), avg_volume])
        
        # Price
        close = np.nan_to_num(df["close"].values)
        
        # Warmup
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Identify breakout conditions
        price_above_upper = close > upper
        price_below_lower = close < lower
        
        # Supertrend direction
        st_up = direction > 0
        st_down = direction < 0
        
        # Volume confirmation
        vol_confirmed = volume > (1.5 * avg_volume)
        
        # Entry conditions
        long_entry = price_above_upper & st_up & vol_confirmed
        short_entry = price_below_lower & st_down & vol_confirmed
        
        # Set signals
        long_signal = pd.Series(0.0, index=df.index)
        short_signal = pd.Series(0.0, index=df.index)
        
        long_signal[long_entry] = 1.0
        short_signal[short_entry] = -1.0
        
        # Combine signals
        signals = long_signal + short_signal
        
        return signals