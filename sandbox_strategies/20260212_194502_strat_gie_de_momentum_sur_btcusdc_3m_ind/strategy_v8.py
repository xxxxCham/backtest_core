from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_momentum_three_indicator")

    @property
    def required_indicators(self) -> List[str]:
        return ["macd", "roc", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"macd_fast": 12, "macd_signal": 9, "macd_slow": 26, "roc_period": 10, "rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 2.0, "tp_atr_mult": 4.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "macd_fast": ParameterSpec(5, 20, 1),
            "macd_signal": ParameterSpec(5, 15, 1),
            "macd_slow": ParameterSpec(20, 50, 1),
            "roc_period": ParameterSpec(5, 20, 1),
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(10, 30, 1),
            "rsi_period": ParameterSpec(5, 20, 1),
            "stop_atr_mult": ParameterSpec(1.0, 4.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 8.0, 0.1),
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
        macd = indicators["macd"]
        roc = np.nan_to_num(indicators["roc"])
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Compute histogram difference
        hist = np.nan_to_num(macd["histogram"])
        hist_diff = np.diff(hist, prepend=np.nan)
        
        # Entry conditions
        long_condition = (hist > 0) & (hist_diff > 0) & (roc > 0) & (rsi < params["rsi_overbought"])
        short_condition = (hist < 0) & (hist_diff < 0) & (roc < 0) & (rsi > params["rsi_oversold"])
        
        # Exit conditions
        exit_long = (hist < 0) & (roc < 0)
        exit_short = (hist > 0) & (roc > 0)
        
        # Convert to boolean masks
        long_mask = long_condition
        short_mask = short_condition
        exit_long_mask = exit_long
        exit_short_mask = exit_short
        
        # Generate signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals