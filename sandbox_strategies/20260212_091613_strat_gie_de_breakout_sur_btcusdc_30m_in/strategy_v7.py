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
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                value_range=(1.0, 3.0),
                default=1.5,
                step=0.1
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                value_range=(2.0, 6.0),
                default=3.5,
                step=0.1
            ),
            "warmup": ParameterSpec(
                name="warmup",
                value_range=(30, 100),
                default=50,
                step=10
            )
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
        
        # Safe access to bollinger bands
        upper_band = np.nan_to_num(bb["upper"])
        lower_band = np.nan_to_num(bb["lower"])
        middle_band = np.nan_to_num(bb["middle"])
        
        # Safe access to supertrend
        supertrend_value = np.nan_to_num(st["supertrend"])
        supertrend_direction = np.nan_to_num(st["direction"])
        
        # Volume and BB width calculations
        close = np.nan_to_num(df["close"].values)
        volume = np.nan_to_num(df["volume"].values)
        bb_width = upper_band - lower_band
        bb_mean_width = pd.Series(bb_width).rolling(window=20).mean().values
        
        # Average volume
        avg_volume = pd.Series(volume).rolling(window=20).mean().values
        
        # BB contraction detection
        bb_contracting = bb_width < pd.Series(bb_width).rolling(window=20).mean().values
        
        # Entry conditions
        long_condition = (close > upper_band) & (supertrend_direction > 0) & (bb_width > bb_mean_width * 1.5) & (volume > avg_volume * 1.2)
        short_condition = (close < lower_band) & (supertrend_direction < 0) & (bb_width > bb_mean_width * 1.5) & (volume > avg_volume * 1.2)
        
        # Exit conditions
        exit_long = (close < upper_band) & (supertrend_direction < 0) & (bb_contracting == True)
        exit_short = (close > lower_band) & (supertrend_direction > 0) & (bb_contracting == True)
        
        # Generate signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0
        signals[exit_long] = 0.0
        signals[exit_short] = 0.0
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals