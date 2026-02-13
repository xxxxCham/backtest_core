from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_supertrend_atr_breakout_with_volume_filter")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "supertrend", "atr", "volume_oscillator"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "stop_atr_mult": ParameterSpec("stop_atr_mult", 1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec("tp_atr_mult", 2.0, 6.0, 0.1),
            "warmup": ParameterSpec("warmup", 20, 100, 5)
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
        upper_band = np.nan_to_num(bb["upper"])
        lower_band = np.nan_to_num(bb["lower"])
        close = np.nan_to_num(df["close"].values)
        st = indicators["supertrend"]
        supertrend_direction = np.nan_to_num(st["direction"])
        atr = np.nan_to_num(indicators["atr"])
        volume_oscillator = np.nan_to_num(indicators["volume_oscillator"])
        
        # Entry conditions
        long_condition = (close > upper_band) & (supertrend_direction > 0) & (volume_oscillator > 0)
        short_condition = (close < lower_band) & (supertrend_direction < 0) & (volume_oscillator < 0)
        
        # Exit conditions
        exit_long = (close < upper_band) & (supertrend_direction < 0)
        exit_short = (close > lower_band) & (supertrend_direction > 0)
        
        # Generate signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0
        
        # Apply exit signals
        exit_mask_long = exit_long & (np.roll(signals, 1) == 1.0)
        exit_mask_short = exit_short & (np.roll(signals, 1) == -1.0)
        
        signals[exit_mask_long] = 0.0
        signals[exit_mask_short] = 0.0
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals