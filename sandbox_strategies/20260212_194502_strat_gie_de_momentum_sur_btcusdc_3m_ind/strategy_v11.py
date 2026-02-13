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
            "stop_atr_mult": ParameterSpec(1.0, 4.0, 0.5),
            "tp_atr_mult": ParameterSpec(3.0, 7.0, 0.5),
            "warmup": ParameterSpec(30, 100, 10),
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
        roc_values = np.nan_to_num(indicators["roc"])
        atr_values = np.nan_to_num(indicators["atr"])
        
        # Compute MACD histogram and its difference
        macd_hist = np.nan_to_num(macd["histogram"])
        macd_hist_diff = np.diff(macd_hist, prepend=0)
        
        # Entry conditions
        long_condition = (macd_hist > 0) & (macd_hist_diff > 0) & (roc_values > 0)
        short_condition = (macd_hist < 0) & (macd_hist_diff < 0) & (roc_values < 0)
        
        # Exit conditions
        exit_long = (macd_hist < 0) & (roc_values < 0)
        exit_short = (macd_hist > 0) & (roc_values > 0)
        
        # Set signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0
        
        # Handle exits
        for i in range(1, len(signals)):
            if signals.iloc[i-1] == 1.0 and exit_long[i]:
                signals.iloc[i] = 0.0
            elif signals.iloc[i-1] == -1.0 and exit_short[i]:
                signals.iloc[i] = 0.0
                
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals