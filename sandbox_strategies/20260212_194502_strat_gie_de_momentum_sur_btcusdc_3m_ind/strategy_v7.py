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
        return ["macd", "roc", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"macd_fast": 12, "macd_signal": 9, "macd_slow": 26, "roc_period": 10, "stop_atr_mult": 2.0, "tp_atr_mult": 4.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "macd_fast": ParameterSpec(5, 20, 1),
            "macd_signal": ParameterSpec(5, 15, 1),
            "macd_slow": ParameterSpec(20, 40, 1),
            "roc_period": ParameterSpec(5, 20, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(3.0, 6.0, 0.1),
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
        roc = indicators["roc"]
        atr = indicators["atr"]
        
        # Compute MACD histogram
        macd_hist = np.nan_to_num(macd["histogram"])
        
        # Compute ROC values
        roc_values = np.nan_to_num(roc)
        
        # Compute ROC moving average
        roc_period = int(params.get("roc_period", 10))
        roc_ma = np.full_like(roc_values, np.nan)
        for i in range(roc_period, len(roc_values)):
            roc_ma[i] = np.mean(roc_values[i - roc_period:i])
        roc_ma = np.nan_to_num(roc_ma)
        
        # Entry conditions
        # Long entry: MACD histogram positive and increasing, ROC positive, ROC above its MA
        long_condition = (
            (macd_hist > 0) &
            (np.diff(macd_hist, prepend=np.nan) > 0) &
            (roc_values > 0) &
            (roc_values > roc_ma)
        )
        
        # Short entry: MACD histogram negative and decreasing, ROC negative, ROC below its MA
        short_condition = (
            (macd_hist < 0) &
            (np.diff(macd_hist, prepend=np.nan) < 0) &
            (roc_values < 0) &
            (roc_values < roc_ma)
        )
        
        # Convert conditions to boolean masks
        long_mask = pd.Series(long_condition, index=df.index, dtype=bool)
        short_mask = pd.Series(short_condition, index=df.index, dtype=bool)
        
        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals