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
            "warmup": ParameterSpec(30, 100, 1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Extract indicators
        macd = indicators["macd"]
        roc = indicators["roc"]
        atr = indicators["atr"]
        
        # Compute MACD histogram
        macd_histogram = np.nan_to_num(macd["macd"] - macd["signal"])
        
        # Compute ROC values
        roc_values = np.nan_to_num(roc)
        
        # Compute ATR values
        atr_values = np.nan_to_num(atr)
        
        # Compute momentum acceleration
        macd_hist_diff = np.diff(macd_histogram, prepend=0)
        roc_positive = roc_values > 0
        roc_negative = roc_values < 0
        
        # Long entry: MACD histogram positive and increasing, ROC positive
        long_condition = (macd_histogram > 0) & (macd_hist_diff > 0) & roc_positive
        
        # Short entry: MACD histogram negative and decreasing, ROC negative
        short_condition = (macd_histogram < 0) & (macd_hist_diff < 0) & roc_negative
        
        # Create signal series
        long_signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        short_signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Assign signals
        long_signals[long_condition] = 1.0
        short_signals[short_condition] = -1.0
        
        # Combine signals
        signals = long_signals + short_signals
        
        return signals