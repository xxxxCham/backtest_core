from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="momentum_btcusdc_30m_short_revised")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "mfi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "stop_atr_mult": ParameterSpec(1.0, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 4.0, 0.1),
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
        
        rsi_overbought = params["rsi_overbought"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        warmup = int(params.get("warmup", 50))
        
        signals.iloc[:warmup] = 0.0
        
        # Entry condition: RSI crosses above overbought threshold with MFI confirmation
        rsi_crossed = (rsi[1:] > rsi_overbought) & (rsi[:-1] <= rsi_overbought)
        mfi_confirmed = (mfi[1:] < 50) & (mfi[:-1] >= 50)
        
        entry_mask = rsi_crossed & mfi_confirmed
        
        # Short signal
        short_signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        short_signals.iloc[1:] = np.where(entry_mask, -1.0, 0.0)
        
        signals = short_signals
        
        return signals