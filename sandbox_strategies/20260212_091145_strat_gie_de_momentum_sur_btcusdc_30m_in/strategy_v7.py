from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="momentum_btcusdc_30m_revised")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "macd", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 2.0, "tp_atr_mult": 4.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(50, 90, 1),
            "rsi_oversold": ParameterSpec(10, 50, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 10.0, 0.1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        rsi = np.nan_to_num(indicators["rsi"])
        macd = indicators["macd"]
        macd_histogram = np.nan_to_num(macd["histogram"])
        atr = np.nan_to_num(indicators["atr"])
        
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        warmup = int(params.get("warmup", 50))
        
        signals.iloc[:warmup] = 0.0
        
        rsi_prev = np.roll(rsi, 1)
        rsi_prev[0] = rsi[0]
        
        # Entry long conditions
        long_entry_cond1 = (rsi > rsi_overbought) & (macd_histogram > 0) & (rsi_prev <= rsi_overbought)
        long_entry_cond2 = (rsi > 50) & (rsi > rsi_oversold + 10)
        long_entry = long_entry_cond1 & long_entry_cond2
        
        # Entry short conditions
        short_entry_cond1 = (rsi < rsi_oversold) & (macd_histogram < 0) & (rsi_prev >= rsi_oversold)
        short_entry_cond2 = (rsi < 50) & (rsi < rsi_overbought - 10)
        short_entry = short_entry_cond1 & short_entry_cond2
        
        # Exit conditions
        exit_cond1 = (rsi > rsi_overbought + 5) | (rsi < rsi_oversold - 5)
        exit_cond2 = ((rsi > 50) & (rsi_prev <= 50)) | ((rsi < 50) & (rsi_prev >= 50))
        exit_signal = exit_cond1 | exit_cond2
        
        # Generate signals
        long_signals = pd.Series(0.0, index=df.index)
        short_signals = pd.Series(0.0, index=df.index)
        
        long_signals[long_entry] = 1.0
        short_signals[short_entry] = -1.0
        
        # Apply exit conditions
        long_exit = exit_signal & (long_signals == 1.0)
        short_exit = exit_signal & (short_signals == -1.0)
        
        long_signals[long_exit] = 0.0
        short_signals[short_exit] = 0.0
        
        signals = long_signals + short_signals
        
        return signals