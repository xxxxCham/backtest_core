from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="trend_following_btcusdc_30m_adx_sma_atr_revised")

    @property
    def required_indicators(self) -> List[str]:
        return ["sma", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"adx_period": 14, "adx_threshold": 25.0, "sma_period": 50, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 100}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec("adx_period", 5, 30, 1),
            "adx_threshold": ParameterSpec("adx_threshold", 10.0, 40.0, 1.0),
            "sma_period": ParameterSpec("sma_period", 20, 100, 5),
            "stop_atr_mult": ParameterSpec("stop_atr_mult", 1.0, 5.0, 0.5),
            "tp_atr_mult": ParameterSpec("tp_atr_mult", 2.0, 10.0, 1.0),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Extract indicators
        sma = np.nan_to_num(indicators["sma"])
        adx = np.nan_to_num(indicators["adx"]["adx"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Entry conditions
        close = np.nan_to_num(df["close"].values)
        sma_shifted = np.roll(sma, 1)
        adx_threshold = params.get("adx_threshold", 25.0)
        
        entry_long = (close > sma) & (adx > adx_threshold) & (sma > sma_shifted)
        
        # Exit conditions
        exit_long = (sma < sma_shifted) | (adx < 20.0)
        
        # Find entry and exit points
        entry_indices = np.where(entry_long)[0]
        exit_indices = np.where(exit_long)[0]
        
        # Create signal series with long entries
        for i in entry_indices:
            if i > 0:
                signals.iloc[i] = 1.0
        
        # Apply exit logic
        for i in range(len(signals)):
            if signals.iloc[i] == 1.0:
                # Check if exit condition is met
                if i < len(exit_indices) and exit_indices[0] > i:
                    signals.iloc[i] = 0.0
        
        return signals