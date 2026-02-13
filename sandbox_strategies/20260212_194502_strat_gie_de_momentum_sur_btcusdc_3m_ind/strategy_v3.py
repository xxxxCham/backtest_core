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
        return {"stop_atr_mult": 2.0, "tp_atr_mult": 4.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "stop_atr_mult": ParameterSpec(param_name="stop_atr_mult", param_type="float", min_value=1.0, max_value=5.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_name="tp_atr_mult", param_type="float", min_value=3.0, max_value=8.0, step=0.5),
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
        atr = np.nan_to_num(indicators["atr"])
        
        # Prepare signals
        macd_macd = np.nan_to_num(macd["macd"])
        macd_signal = np.nan_to_num(macd["signal"])
        
        # Entry conditions
        long_condition = (macd_macd > macd_signal) & (macd_macd > 0) & (roc > 0)
        short_condition = (macd_macd < macd_signal) & (macd_macd < 0) & (roc < 0)
        
        # Exit conditions
        exit_long_condition = macd_macd < macd_signal
        exit_short_condition = macd_macd > macd_signal
        
        # Initialize entry and exit arrays
        entry_long = np.zeros_like(long_condition, dtype=bool)
        entry_short = np.zeros_like(short_condition, dtype=bool)
        exit_long = np.zeros_like(exit_long_condition, dtype=bool)
        exit_short = np.zeros_like(exit_short_condition, dtype=bool)
        
        # Set entry signals
        entry_long[long_condition] = True
        entry_short[short_condition] = True
        
        # Set exit signals
        exit_long[exit_long_condition] = True
        exit_short[exit_short_condition] = True
        
        # Generate signals
        signals[entry_long] = 1.0
        signals[entry_short] = -1.0
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals