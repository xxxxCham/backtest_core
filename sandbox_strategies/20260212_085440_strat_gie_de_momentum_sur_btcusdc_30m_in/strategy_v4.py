from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_30m_momentum_rsi_bollinger_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(20, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 6.0, 0.1),
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
        bb = indicators["bollinger"]
        bb_upper = np.nan_to_num(bb["upper"])
        bb_middle = np.nan_to_num(bb["middle"])
        bb_lower = np.nan_to_num(bb["lower"])
        atr = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)
        
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        warmup = int(params.get("warmup", 50))
        
        signals.iloc[:warmup] = 0.0
        
        # Entry conditions
        rsi_condition = (rsi > rsi_oversold) & (rsi > np.roll(rsi, 1))
        price_condition = (close < bb_lower) & (close > np.roll(close, 1))
        entry_signal = rsi_condition & price_condition
        
        # Exit conditions
        exit_condition = (rsi < rsi_overbought) & (rsi < np.roll(rsi, 1)) & (rsi < 50)
        
        # Generate signals
        entry_indices = np.where(entry_signal)[0]
        exit_indices = np.where(exit_condition)[0]
        
        for i in entry_indices:
            if i > 0 and i < len(signals) - 1:
                signals.iloc[i] = 1.0  # LONG signal
                
        # Apply exit signals
        for i in exit_indices:
            if i > 0 and i < len(signals) - 1:
                if signals.iloc[i-1] == 1.0:
                    signals.iloc[i] = 0.0  # FLAT signal
                    
        return signals