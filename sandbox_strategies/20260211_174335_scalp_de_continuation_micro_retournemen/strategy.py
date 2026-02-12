from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="scalp_crypto_ema_rsi_bollinger")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi", "bollinger"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(min_val=5, max_val=50, default=14, param_type="int"),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Get indicators and handle NaNs
        ema_val = np.nan_to_num(indicators["ema"])
        rsi_val = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        upper = np.nan_to_num(bb["upper"])
        lower = np.nan_to_num(bb["lower"])
        
        close = df["close"].values
        
        # Entry conditions
        long_entry = (close > ema_val) & (rsi_val < params["rsi_overbought"]) & (
            (close > upper) | (close < lower)
        )
        short_entry = (close < ema_val) & (rsi_val > params["rsi_oversold"]) & (
            (close > upper) | (close < lower)
        )
        
        # Exit conditions
        is_long = signals == 1.0
        is_short = signals == -1.0
        
        long_exit = (close < lower) | (rsi_val > params["rsi_overbought"])
        short_exit = (close > upper) | (rsi_val < params["rsi_oversold"])
        
        # Update signals
        signals[long_entry] = 1.0
        signals[short_entry] = -1.0
        
        signals[is_long & long_exit] = 0.0
        signals[is_short & short_exit] = 0.0
        
        # Warmup period
        warmup = 50
        signals.iloc[:warmup] = 0.0
        
        return signals