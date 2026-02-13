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
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(10, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 6.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        rsi = np.nan_to_num(indicators["rsi"])
        macd = indicators["macd"]
        macd_histogram = np.nan_to_num(macd["histogram"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Entry conditions
        long_condition = (rsi > params["rsi_oversold"]) & (macd_histogram > 0) & (macd_histogram > np.roll(macd_histogram, 1))
        short_condition = (rsi < params["rsi_overbought"]) & (macd_histogram < 0) & (macd_histogram < np.roll(macd_histogram, 1))
        
        # Exit conditions
        exit_long_condition = (rsi > params["rsi_overbought"]) | (rsi < params["rsi_oversold"]) | ((macd_histogram < 0) & (rsi > 50)) | ((macd_histogram > 0) & (rsi < 50))
        exit_short_condition = (rsi > params["rsi_overbought"]) | (rsi < params["rsi_oversold"]) | ((macd_histogram < 0) & (rsi > 50)) | ((macd_histogram > 0) & (rsi < 50))
        
        # Set signals
        long_entries = long_condition & ~np.roll(long_condition, 1)
        short_entries = short_condition & ~np.roll(short_condition, 1)
        
        signals[long_entries] = 1.0
        signals[short_entries] = -1.0
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals