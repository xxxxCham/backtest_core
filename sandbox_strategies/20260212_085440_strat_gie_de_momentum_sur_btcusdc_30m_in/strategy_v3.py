from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_30m_momentum_macd_rsi_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "macd", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 2.0, "tp_atr_mult": 4.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(20, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(3.0, 6.0, 0.1),
            "warmup": ParameterSpec(30, 100, 1),
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
        price = np.nan_to_num(df["close"].values)
        
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 4.5)
        warmup = int(params.get("warmup", 50))
        
        # Entry conditions
        entry_long = (
            (rsi > rsi_oversold) &
            (rsi > np.roll(rsi, 1)) &
            (macd_histogram > 0) &
            (macd_histogram > np.roll(macd_histogram, 1))
        )
        
        # Exit conditions
        exit_long = (
            ((rsi < rsi_overbought) & (rsi < np.roll(rsi, 1)) & (rsi < 50)) |
            ((rsi > 70) & (price < np.roll(price, 1)))
        )
        
        # Generate signals
        entry_mask = entry_long
        exit_mask = exit_long
        
        # Initialize signals
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Set entry signals
        entry_indices = np.where(entry_mask)[0]
        for idx in entry_indices:
            if idx > 0:
                signals.iloc[idx] = 1.0  # LONG
                
        # Set exit signals
        exit_indices = np.where(exit_mask)[0]
        for idx in exit_indices:
            if idx > 0:
                if signals.iloc[idx-1] == 1.0:
                    signals.iloc[idx] = 0.0  # FLAT
                    
        # Set warmup period
        signals.iloc[:warmup] = 0.0
        
        return signals