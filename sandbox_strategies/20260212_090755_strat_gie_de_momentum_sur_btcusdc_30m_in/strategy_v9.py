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
        return ["rsi", "mfi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 2.0, "tp_atr_mult": 4.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(10, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 10.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1),
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
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 4.5)
        warmup = int(params.get("warmup", 50))
        
        signals.iloc[:warmup] = 0.0
        
        # Entry condition for short: RSI overbought with MFI confirmation
        entry_condition = (rsi > rsi_overbought) & (mfi > 80)
        
        # Confirmation: RSI and MFI are decreasing
        rsi_prev = np.roll(rsi, 1)
        mfi_prev = np.roll(mfi, 1)
        confirmation_condition = (rsi < rsi_prev) & (mfi < mfi_prev)
        
        # Combine entry conditions
        entry_signal = entry_condition & confirmation_condition
        
        # Exit condition: RSI oversold OR ATR increased by 20% OR RSI and MFI both above 50
        exit_condition = (rsi < rsi_oversold) | (atr > np.roll(atr, 1) * 1.2) | ((rsi > 50) & (mfi > 50))
        
        # Generate short signals
        short_positions = np.zeros_like(rsi, dtype=float)
        short_positions[entry_signal] = -1.0
        short_positions[exit_condition] = 0.0
        
        # Handle position management: only allow one short at a time
        in_position = False
        for i in range(len(short_positions)):
            if short_positions[i] == -1.0:
                in_position = True
            elif short_positions[i] == 0.0 and in_position:
                in_position = False
            elif in_position and i > 0:
                short_positions[i] = -1.0
            else:
                short_positions[i] = 0.0
        
        signals = pd.Series(short_positions, index=df.index, dtype=np.float64)
        
        return signals