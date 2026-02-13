from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_momentum_short_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "macd", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(10, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1)
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
        
        # Short entry condition: RSI > 70 and MACD histogram > 0
        entry_short = (rsi > rsi_overbought) & (macd_hist > 0)
        
        # Exit condition: RSI < 50 or RSI > 70 and MACD histogram < 0
        exit_short = (rsi < (rsi_overbought + rsi_oversold) / 2) | ((rsi > rsi_overbought) & (macd_hist < 0))
        
        # Generate signals
        entry_mask = entry_short
        exit_mask = exit_short
        
        # Short signal
        short_signal = pd.Series(0.0, index=df.index)
        short_signal[entry_mask] = -1.0
        short_signal[exit_mask] = 0.0
        
        signals = short_signal
        
        return signals