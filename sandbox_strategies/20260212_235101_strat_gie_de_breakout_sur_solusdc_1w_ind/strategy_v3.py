from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_breakout_atr_trail_v3")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "atr", "adx"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "adx_period": 14,
            "adx_threshold": 25,
            "atr_period": 14,
            "bollinger_period": 20,
            "bollinger_std_dev": 2,
            "leverage": 1,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.5,
            "warmup": 50
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(type=int, default=14, min_val=1, max_val=50, description="ADX period"),
            "adx_threshold": ParameterSpec(type=int, default=25, min_val=0, max_val=100, description="Minimum ADX value for breakout confirmation"),
            "atr_period": ParameterSpec(type=int, default=14, min_val=1, max_val=50, description="ATR period"),
            "bollinger_period": ParameterSpec(type=int, default=20, min_val=10, max_val=50, description="Bollinger Bands period"),
            "bollinger_std_dev": ParameterSpec(type=float, default=2.0, min_val=0.5, max_val=5.0, description="Standard deviation multiplier for Bollinger Bands"),
            "leverage": ParameterSpec(type=int, default=1, min_val=1, max_val=3, description="Trading leverage"),
            "stop_atr_mult": ParameterSpec(type=float, default=1.5, min_val=0.1, max_val=10.0, description="ATR multiplier for stop loss"),
            "tp_atr_mult": ParameterSpec(type=float, default=3.5, min_val=0.1, max_val=10.0, description="ATR multiplier for take profit"),
            "warmup": ParameterSpec(type=int, default=50, min_val=0, max_val=200, description="Warmup period without signals")
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Calculate median volume for volume confirmation
        volume = df["volume"].values
        median_volume = np.median(volume)
        
        # Extract and sanitize indicator values
        adx = np.nan_to_num(indicators["adx"]["adx"])
        atr = np.nan_to_num(indicators["atr"])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        close = df["close"].values
        volume_arr = volume
        
        entry_long_mask = (
            (close > bb_upper) &
            (volume_arr > median_volume) &
            (adx > self.default_params["adx_threshold"])
        )
        
        entry_short_mask = (
            (close < bb_lower) &
            (volume_arr > median_volume) &
            (adx > self.default_params["adx_threshold"])
        )
        
        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        
        # Apply ATR-based risk levels on entry signals
        atr_values = np.nan_to_num(indicators["atr"])
        
        long_entries = entry_long_mask
        short_entries = entry_short_mask
        
        if long_entries.any():
            entry_points = close[long_entries]
            df.loc[long_entries, "bb_stop_long"] = entry_points - self.default_params["stop_atr_mult"] * atr_values[long_entries]
            df.loc[long_entries, "bb_tp_long"] = entry_points + self.default_params["tp_atr_mult"] * atr_values[long_entries]
        
        if short_entries.any():
            entry_points = close[short_entries]
            df.loc[short_entries, "bb_stop_short"] = entry_points + self.default_params["stop_atr_mult"] * atr_values[short_entries]
            df.loc[short_entries, "bb_tp_short"] = entry_points - self.default_params["tp_atr_mult"] * atr_values[short_entries]
        
        # Apply warmup masking
        warmup = self.default_params["warmup"]
        signals.iloc[:warmup] = 0.0
        
        return signals