from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="rsi_macd_atr_momentum")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "macd", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(param_name="rsi_overbought", param_type="int", min_value=50, max_value=90, step=5),
            "rsi_oversold": ParameterSpec(param_name="rsi_oversold", param_type="int", min_value=10, max_value=50, step=5),
            "rsi_period": ParameterSpec(param_name="rsi_period", param_type="int", min_value=5, max_value=30, step=5),
            "stop_atr_mult": ParameterSpec(param_name="stop_atr_mult", param_type="float", min_value=1.0, max_value=3.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_name="tp_atr_mult", param_type="float", min_value=2.0, max_value=5.0, step=0.5),
            "warmup": ParameterSpec(param_name="warmup", param_type="int", min_value=30, max_value=100, step=10),
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
        macd_hist = np.nan_to_num(macd["histogram"])
        atr = np.nan_to_num(indicators["atr"])
        
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))
        
        signals.iloc[:warmup] = 0.0
        
        # Entry long: RSI crosses above oversold with positive MACD histogram
        entry_long = (rsi > rsi_oversold) & (rsi > np.roll(rsi, 1)) & (macd_hist > 0) & (rsi < rsi_overbought)
        
        # Entry short: RSI crosses below overbought with negative MACD histogram
        entry_short = (rsi < rsi_overbought) & (rsi < np.roll(rsi, 1)) & (macd_hist < 0) & (rsi > rsi_oversold)
        
        # Exit conditions
        exit_long = (rsi > rsi_overbought) | (rsi < rsi_oversold) | ((rsi > np.roll(rsi, 1)) & (macd_hist < 0))
        exit_short = (rsi < rsi_oversold) | (rsi > rsi_overbought) | ((rsi < np.roll(rsi, 1)) & (macd_hist > 0))
        
        # Generate signals
        long_entries = np.where(entry_long, 1.0, 0.0)
        short_entries = np.where(entry_short, -1.0, 0.0)
        long_exits = np.where(exit_long, 0.0, 1.0)
        short_exits = np.where(exit_short, 0.0, 1.0)
        
        # Combine entries and exits
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        signals.iloc[warmup:] = np.where(long_entries[warmup:] == 1.0, 1.0, 
                                         np.where(short_entries[warmup:] == -1.0, -1.0, 
                                                 np.where(long_exits[warmup:] == 0.0, 0.0, 
                                                        np.where(short_exits[warmup:] == 0.0, 0.0, signals.iloc[warmup:]))))
        
        return signals