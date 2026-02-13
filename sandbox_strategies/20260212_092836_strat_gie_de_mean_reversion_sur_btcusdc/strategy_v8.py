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
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(50, 100, 1, "Overbought RSI level"),
            "rsi_oversold": ParameterSpec(0, 50, 1, "Oversold RSI level"),
            "rsi_period": ParameterSpec(5, 30, 1, "RSI period"),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1, "Stop-loss multiplier based on ATR"),
            "tp_atr_mult": ParameterSpec(2.0, 6.0, 0.1, "Take-profit multiplier based on ATR"),
            "warmup": ParameterSpec(20, 100, 1, "Warmup bars to skip signal generation")
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        price = np.nan_to_num(df['close'].values)
        bb = indicators["bollinger"]
        lower_bb = np.nan_to_num(bb["lower"])
        middle_bb = np.nan_to_num(bb["middle"])
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Extract params
        rsi_overbought = params.get("rsi_overbought", 70)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))
        
        # Entry condition: price touches or crosses below lower Bollinger band AND RSI is overbought
        entry_condition = (price <= lower_bb) & (rsi > rsi_overbought)
        
        # Exit condition: price crosses back above middle Bollinger band
        exit_condition = price >= middle_bb
        
        # Generate signals
        entry_indices = np.where(entry_condition)[0]
        exit_indices = np.where(exit_condition)[0]
        
        # Initialize signal array
        signal_values = np.zeros_like(price)
        
        # Apply short signals
        for i in entry_indices:
            if i < len(signal_values):
                signal_values[i] = -1.0  # Short signal
        
        # Apply exit signals
        for i in exit_indices:
            if i < len(signal_values) and signal_values[i] == -1.0:
                signal_values[i] = 0.0  # Flat signal on exit
        
        # Convert to pandas Series
        signals = pd.Series(signal_values, index=df.index, dtype=np.float64)
        
        # Warmup protection
        signals.iloc[:warmup] = 0.0
        
        return signals