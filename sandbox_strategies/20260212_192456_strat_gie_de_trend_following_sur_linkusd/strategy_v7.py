from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="supertrend_adx_atr_trend_following")

    @property
    def required_indicators(self) -> List[str]:
        return ["supertrend", "adx", "atr", "ema"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"adx_period": 14, "adx_threshold": 25, "atr_threshold": 0.005, "ema_period": 20, "stop_atr_mult": 1.0, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(param_type="int", min_value=5, max_value=30, step=1),
            "adx_threshold": ParameterSpec(param_type="float", min_value=10, max_value=50, step=1),
            "atr_threshold": ParameterSpec(param_type="float", min_value=0.001, max_value=0.02, step=0.001),
            "ema_period": ParameterSpec(param_type="int", min_value=10, max_value=50, step=5),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=0.5, max_value=2.0, step=0.5),
            "supertrend_multiplier": ParameterSpec(param_type="float", min_value=1.0, max_value=5.0, step=0.5),
            "supertrend_period": ParameterSpec(param_type="int", min_value=5, max_value=20, step=1),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=4.0, step=0.5),
            "warmup": ParameterSpec(param_type="int", min_value=20, max_value=100, step=10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        supertrend_line = np.nan_to_num(indicators["supertrend"]["supertrend"])
        supertrend_direction = np.nan_to_num(indicators["supertrend"]["direction"])
        adx_value = np.nan_to_num(indicators["adx"]["adx"])
        atr_value = np.nan_to_num(indicators["atr"])
        ema_line = np.nan_to_num(indicators["ema"])
        close = np.nan_to_num(df["close"].values)
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Entry conditions
        # Supertrend confirms uptrend
        trend_up = supertrend_direction > 0
        # Price is above EMA
        price_above_ema = close > ema_line
        # ATR-based volatility threshold
        volatility_ok = atr_value > params["atr_threshold"]
        # Supertrend line below price
        supertrend_confirms = supertrend_line < close
        
        # Entry signal
        entry_signal = (supertrend_confirms) & (trend_up) & (price_above_ema) & (volatility_ok)
        
        # Exit conditions
        # Supertrend line crosses above price
        exit_signal = (supertrend_line > close)
        # ADX strength drops below threshold
        adx_weak = adx_value < params["adx_threshold"]
        
        # Exit signal
        exit_condition = exit_signal | adx_weak
        
        # Generate signals
        entry_indices = np.where(entry_signal)[0]
        exit_indices = np.where(exit_condition)[0]
        
        # Initialize entry and exit arrays
        entry_mask = np.zeros(len(df), dtype=bool)
        exit_mask = np.zeros(len(df), dtype=bool)
        
        # Mark entry points
        entry_mask[entry_indices] = True
        
        # Mark exit points
        exit_mask[exit_indices] = True
        
        # Long signals
        signals[entry_mask] = 1.0
        
        # Flatten positions on exit
        signals[exit_mask] = 0.0
        
        return signals