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
            "macd_slow": ParameterSpec(20, 50, 1),
            "roc_period": ParameterSpec(5, 20, 1),
            "stop_atr_mult": ParameterSpec(1.0, 4.0, 0.5),
            "tp_atr_mult": ParameterSpec(3.0, 8.0, 0.5),
            "warmup": ParameterSpec(20, 100, 10),
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
        macd_hist = np.nan_to_num(indicators["macd"]["histogram"])
        roc = np.nan_to_num(indicators["roc"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Entry conditions
        # Long entry: MACD histogram positive and accelerating, ROC positive
        long_condition = (macd_hist > 0) & (np.diff(macd_hist, prepend=np.nan) > 0) & (roc > 0)
        
        # Short entry: MACD histogram negative and decelerating, ROC negative
        short_condition = (macd_hist < 0) & (np.diff(macd_hist, prepend=np.nan) < 0) & (roc < 0)
        
        # Exit conditions
        # Exit long: MACD histogram becomes negative or ROC becomes negative
        exit_long = (macd_hist < 0) & (roc < 0)
        
        # Exit short: MACD histogram becomes positive or ROC becomes positive
        exit_short = (macd_hist > 0) & (roc > 0)
        
        # Generate signals
        long_entries = np.where(long_condition, 1.0, 0.0)
        short_entries = np.where(short_condition, -1.0, 0.0)
        exit_signals = np.where(exit_long | exit_short, 0.0, 0.0)
        
        # Apply signals
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        signals.iloc[warmup:] = 0.0
        
        # Initialize positions
        position = 0.0
        for i in range(warmup, len(signals)):
            if position == 0:
                if long_entries[i] == 1.0:
                    signals.iloc[i] = 1.0
                    position = 1.0
                elif short_entries[i] == -1.0:
                    signals.iloc[i] = -1.0
                    position = -1.0
            else:
                # Exit condition
                if (position == 1.0 and exit_long[i] == True) or (position == -1.0 and exit_short[i] == True):
                    signals.iloc[i] = 0.0
                    position = 0.0
                else:
                    signals.iloc[i] = position
                    
        return signals