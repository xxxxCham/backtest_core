from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="donchian_rsi_mean_reversion")

    @property
    def required_indicators(self) -> List[str]:
        return ["donchian", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 90, 1),
            "rsi_oversold": ParameterSpec(10, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(1.5, 5.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        donchian = indicators["donchian"]
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Extract donchian bands
        donchian_upper = np.nan_to_num(donchian["upper"])
        donchian_lower = np.nan_to_num(donchian["lower"])
        donchian_middle = np.nan_to_num(donchian["middle"])
        
        # Extract params
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))
        
        # Entry condition: price touches lower band with oversold RSI
        entry_long = (df['close'].values <= donchian_lower) & (rsi < rsi_oversold) & (df['close'].values > df['open'].values)
        
        # Exit condition: price crosses middle band
        exit_long = df['close'].values >= donchian_middle
        
        # Initialize entry and exit signals
        entry_signal = pd.Series(0.0, index=df.index)
        exit_signal = pd.Series(0.0, index=df.index)
        
        # Set entry signals
        entry_signal[entry_long] = 1.0
        
        # Set exit signals
        exit_signal[exit_long] = -1.0
        
        # Combine signals
        signals = entry_signal + exit_signal
        
        # Apply warmup
        signals.iloc[:warmup] = 0.0
        
        # Ensure only long signals
        signals = signals.where(signals > 0, 0.0)
        
        return signals