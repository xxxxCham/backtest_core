from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="trend_following_btcusdc_30m")

    @property
    def required_indicators(self) -> List[str]:
        return ["sma", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"adx_period": 14, "adx_threshold": 25.0, "sma_period": 50, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 100}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(10, 30, 14),
            "adx_threshold": ParameterSpec(10.0, 40.0, 25.0),
            "sma_period": ParameterSpec(20, 100, 50),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 2.0),
            "tp_atr_mult": ParameterSpec(2.0, 10.0, 5.0),
            "warmup": ParameterSpec(50, 200, 100)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        adx_period = int(params.get("adx_period", 14))
        adx_threshold = float(params.get("adx_threshold", 25.0))
        sma_period = int(params.get("sma_period", 50))
        stop_atr_mult = float(params.get("stop_atr_mult", 2.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 5.0))
        warmup = int(params.get("warmup", 100))
        
        sma = np.nan_to_num(indicators["sma"])
        adx = np.nan_to_num(indicators["adx"]["adx"])
        atr = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)
        
        # Entry conditions
        entry_long = (close > sma) & (adx > adx_threshold)
        
        # Exit condition - divergence between SMA and ADX
        sma_diff = sma - np.roll(sma, 1)
        adx_diff = adx - np.roll(adx, 1)
        exit_condition = sma_diff < adx_diff
        
        # Initialize entry and exit signals
        entry_long_signal = pd.Series(0.0, index=df.index)
        exit_signal = pd.Series(0.0, index=df.index)
        
        # Set entry signals
        entry_long_signal[entry_long] = 1.0
        
        # Set exit signals
        exit_signal[exit_condition] = -1.0
        
        # Combine signals
        signals = entry_long_signal + exit_signal
        
        # Ensure no overlapping signals
        signals = signals.clip(lower=0.0, upper=1.0)
        
        # Set warmup period
        signals.iloc[:warmup] = 0.0
        
        return signals