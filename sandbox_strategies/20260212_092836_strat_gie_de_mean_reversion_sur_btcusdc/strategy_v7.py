from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_mean_reversion_bollinger_rsi_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(50, 90, 1),
            "rsi_oversold": ParameterSpec(10, 50, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.5),
            "tp_atr_mult": ParameterSpec(2.0, 10.0, 0.5),
            "warmup": ParameterSpec(10, 100, 5),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        rsi_overbought = params.get("rsi_overbought", 70)
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 5.0)
        warmup = int(params.get("warmup", 50))
        
        bb = indicators["bollinger"]
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])
        
        lower_bb = np.nan_to_num(bb["lower"])
        upper_bb = np.nan_to_num(bb["upper"])
        middle_bb = np.nan_to_num(bb["middle"])
        
        price = np.nan_to_num(df["close"].values)
        
        # Entry condition: price touches lower Bollinger band AND RSI > 70
        entry_condition = (price == lower_bb) & (rsi > rsi_overbought)
        
        # Exit condition: price crosses upper Bollinger band (mean reversion)
        exit_condition = price > upper_bb
        
        # Generate short signals
        short_entries = pd.Series(0.0, index=df.index, dtype=np.float64)
        short_entries[entry_condition] = -1.0
        
        # For simplicity, we'll just mark the first entry and exit at next signal
        # In practice, you'd want to track positions and manage TP/SL
        signals.iloc[:warmup] = 0.0
        
        # Apply entry signals
        signals.loc[short_entries == -1.0] = -1.0
        
        # Simple exit on next crossing
        exit_signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        exit_signals[exit_condition] = 1.0
        signals.loc[exit_signals == 1.0] = 0.0
        
        return signals