from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btc_momentum_short_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "mfi", "atr", "bollinger"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 2.0, "tp_atr_mult": 4.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(50, 90, 1),
            "rsi_oversold": ParameterSpec(10, 50, 1),
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
        
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_period = params.get("rsi_period", 14)
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 4.5)
        warmup = int(params.get("warmup", 50))
        
        rsi = np.nan_to_num(indicators["rsi"])
        mfi = np.nan_to_num(indicators["mfi"])
        atr = np.nan_to_num(indicators["atr"])
        bb = indicators["bollinger"]
        lower_bb = np.nan_to_num(bb["lower"])
        close = np.nan_to_num(df["close"])
        
        # Previous values
        rsi_prev = np.roll(rsi, 1)
        mfi_prev = np.roll(mfi, 1)
        
        # Entry conditions for short
        rsi_below_overbought = rsi < rsi_overbought
        mfi_below_overbought = mfi < rsi_overbought
        rsi_dropping = rsi < rsi_prev
        mfi_dropping = mfi < mfi_prev
        price_below_bb = close < lower_bb
        
        entry_condition = (
            rsi_below_overbought 
            & mfi_below_overbought 
            & rsi_dropping 
            & mfi_dropping 
            & price_below_bb
        )
        
        # Generate signals
        signals[entry_condition] = -1.0
        
        # Warmup protection
        signals.iloc[:warmup] = 0.0
        
        return signals