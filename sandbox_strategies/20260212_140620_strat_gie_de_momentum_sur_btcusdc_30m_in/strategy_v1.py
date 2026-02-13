from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="momentum_btcusdc_30m_short")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "mfi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec("rsi_overbought", 60, 80, 1),
            "rsi_oversold": ParameterSpec("rsi_oversold", 20, 40, 1),
            "stop_atr_mult": ParameterSpec("stop_atr_mult", 1.0, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec("tp_atr_mult", 2.0, 4.0, 0.1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        rsi = np.nan_to_num(indicators["rsi"])
        mfi = np.nan_to_num(indicators["mfi"])
        atr = np.nan_to_num(indicators["atr"])
        
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))
        
        signals.iloc[:warmup] = 0.0
        
        # Short entries: RSI crosses below overbought and MFI confirms bearish momentum
        rsi_cross_below_overbought = (rsi[:-1] >= rsi_overbought) & (rsi[1:] < rsi_overbought)
        mfi_bearish = mfi[1:] < 50  # MFI below 50 indicates bearish momentum
        
        short_entry = rsi_cross_below_overbought & mfi_bearish
        short_entry_indices = np.where(short_entry)[0] + 1  # +1 because we compare with previous bar
        
        for idx in short_entry_indices:
            if idx < len(signals):
                signals.iloc[idx] = -1.0  # Short signal
        
        return signals