from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_meanreversion_30m")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "stoch_rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(70, 80, 1, "rsi_overbought"),
            "rsi_oversold": ParameterSpec(20, 30, 1, "rsi_oversold"),
            "rsi_period": ParameterSpec(10, 14, 1, "rsi_period"),
            "stop_atr_mult": ParameterSpec(1.0, 2.0, 0.1, "stop_atr_mult"),
            "tp_atr_mult": ParameterSpec(2.0, 4.0, 0.1, "tp_atr_mult"),
            "warmup": ParameterSpec(30, 50, 1, "warmup"),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        bb = indicators["bollinger"]
        stoch_rsi = indicators["stoch_rsi"]
        atr = np.nan_to_num(indicators["atr"])
        
        # Bollinger bands
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])
        lower = np.nan_to_num(bb["lower"])
        
        # Stochastic RSI
        stoch_rsi_k = np.nan_to_num(stoch_rsi["k"])
        stoch_rsi_d = np.nan_to_num(stoch_rsi["d"])
        
        # Price
        close = np.nan_to_num(df["close"].values)
        
        # Entry conditions for short
        short_entry_cond = (close > upper) & (stoch_rsi_k > 80) & (stoch_rsi_d > 80)
        
        # Exit condition when price crosses middle band
        exit_cond = (close < middle) & (np.roll(close, 1) >= np.roll(middle, 1))
        
        # Generate signals
        short_entry = pd.Series(0.0, index=df.index)
        short_entry[short_entry_cond] = -1.0
        
        # Apply exit condition
        exit_signal = pd.Series(0.0, index=df.index)
        exit_signal[exit_cond] = 0.0
        
        # Combine signals
        signals = short_entry
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals