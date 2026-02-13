from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="momentum_avaxusdc_1d")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "macd", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(50, 90, 1),
            "rsi_oversold": ParameterSpec(10, 50, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        rsi = np.nan_to_num(indicators["rsi"])
        macd = indicators["macd"]
        macd_macd = np.nan_to_num(macd["macd"])
        macd_signal = np.nan_to_num(macd["signal"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Entry conditions
        entry_long = (rsi > rsi_oversold) & (np.roll(rsi, 1) <= rsi_oversold) & (macd_macd > macd_signal) & (np.roll(macd_macd, 1) <= np.roll(macd_signal, 1))
        entry_short = (rsi < rsi_overbought) & (np.roll(rsi, 1) >= rsi_overbought) & (macd_macd < macd_signal) & (np.roll(macd_macd, 1) >= np.roll(macd_signal, 1))
        
        # Exit conditions
        exit_long = (rsi > rsi_overbought) | (rsi < rsi_oversold) | ((rsi - np.roll(rsi, 1) < 0) & (np.roll(rsi, 1) > np.roll(rsi, 2)))
        exit_short = (rsi < rsi_oversold) | (rsi > rsi_overbought) | ((rsi - np.roll(rsi, 1) > 0) & (np.roll(rsi, 1) < np.roll(rsi, 2)))
        
        # Generate signals
        long_entries = entry_long
        short_entries = entry_short
        
        # Set signals
        signals[long_entries] = 1.0
        signals[short_entries] = -1.0
        
        return signals