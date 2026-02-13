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
        return {"stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "stop_atr_mult": ParameterSpec(param_name="stop_atr_mult", param_type="float", min_value=1.0, max_value=3.0, step=0.1),
            "tp_atr_mult": ParameterSpec(param_name="tp_atr_mult", param_type="float", min_value=2.0, max_value=6.0, step=0.1),
            "warmup": ParameterSpec(param_name="warmup", param_type="int", min_value=20, max_value=100, step=10)
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
        
        # Compute signals
        macd_macd = np.nan_to_num(macd["macd"])
        macd_signal = np.nan_to_num(macd["signal"])
        
        # Entry conditions
        long_condition = (macd_macd > macd_signal) & (macd_macd > 0) & (roc > 0) & (rsi < 70)
        short_condition = (macd_macd < macd_signal) & (macd_macd < 0) & (roc < 0) & (rsi > 30)
        
        # Exit conditions
        exit_long_condition = macd_macd < macd_signal
        exit_short_condition = macd_macd > macd_signal
        
        # Create signal array
        long_signals = np.where(long_condition, 1.0, 0.0)
        short_signals = np.where(short_condition, -1.0, 0.0)
        
        # Apply exits
        for i in range(1, len(signals)):
            if signals.iloc[i-1] == 1.0 and exit_long_condition[i]:
                signals.iloc[i] = 0.0
            elif signals.iloc[i-1] == -1.0 and exit_short_condition[i]:
                signals.iloc[i] = 0.0
            else:
                signals.iloc[i] = signals.iloc[i-1]
        
        # Apply entries
        signals = pd.Series(np.where(long_signals == 1.0, 1.0, 
                                   np.where(short_signals == -1.0, -1.0, signals.values)), 
                          index=df.index, dtype=np.float64)
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals