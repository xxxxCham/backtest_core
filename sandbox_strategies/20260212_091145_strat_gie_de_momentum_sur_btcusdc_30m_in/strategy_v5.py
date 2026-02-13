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
            "rsi_overbought": ParameterSpec(param_type="int", min_value=60, max_value=85, step=5),
            "rsi_oversold": ParameterSpec(param_type="int", min_value=15, max_value=40, step=5),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=3.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=3.0, max_value=6.0, step=0.5),
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
        
        rsi_prev = np.roll(rsi, 1)
        rsi_prev[0] = rsi[0]
        
        # Entry long condition
        entry_long_condition = (
            (rsi > rsi_overbought) &
            (macd_histogram > 0) &
            (rsi_prev <= rsi_overbought) &
            (rsi > 50)
        )
        
        # Entry short condition
        entry_short_condition = (
            (rsi < rsi_oversold) &
            (macd_histogram < 0) &
            (rsi_prev >= rsi_oversold) &
            (rsi < 50)
        )
        
        # Exit conditions
        exit_long_condition = (
            (rsi > rsi_overbought + 5) |
            (rsi < rsi_oversold - 5) |
            ((rsi > 50) & (rsi_prev <= 50))
        )
        
        exit_short_condition = (
            (rsi > rsi_overbought + 5) |
            (rsi < rsi_oversold - 5) |
            ((rsi < 50) & (rsi_prev >= 50))
        )
        
        # Generate signals
        long_signals = np.where(entry_long_condition, 1.0, 0.0)
        short_signals = np.where(entry_short_condition, -1.0, 0.0)
        
        # Combine signals
        signals = pd.Series(long_signals + short_signals, index=df.index, dtype=np.float64)
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals