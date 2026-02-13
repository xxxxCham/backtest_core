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
        roc = np.nan_to_num(indicators["roc"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Compute MACD values
        macd_line = np.nan_to_num(macd["macd"])
        signal_line = np.nan_to_num(macd["signal"])
        
        # Compute ROC with proper period
        roc_period = int(params.get("roc_period", 10))
        roc_shifted = np.roll(roc, roc_period)
        roc_diff = roc - roc_shifted
        
        # Entry conditions
        long_condition = (macd_line > signal_line) & (roc > 0) & (roc_diff > 0)
        short_condition = (macd_line < signal_line) & (roc < 0) & (roc_diff < 0)
        
        # Exit conditions
        exit_long_condition = (roc < 0) & (macd_line < signal_line)
        exit_short_condition = (roc > 0) & (macd_line > signal_line)
        
        # Generate signals
        long_signals = np.where(long_condition, 1.0, 0.0)
        short_signals = np.where(short_condition, -1.0, 0.0)
        
        # Exit signals
        exit_long_signals = np.where(exit_long_condition, 0.0, 0.0)
        exit_short_signals = np.where(exit_short_condition, 0.0, 0.0)
        
        # Combine signals
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        signals.iloc[:] = long_signals + short_signals
        
        # Apply warmup
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals