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
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(10, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 2.5, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
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
        
        rsi_prev = np.roll(rsi, 1)
        rsi_prev[0] = rsi[0]
        
        # MACD histogram for confirmation (assuming MACD is available)
        # For this strategy, we'll use a simple approach without explicit MACD
        # Since MACD is not in required indicators, we'll approximate momentum with RSI
        # and price position relative to Bollinger Bands
        
        long_condition = (rsi > rsi_overbought) & (rsi_prev <= rsi_overbought) & (close > bb_upper)
        short_condition = (rsi < rsi_oversold) & (rsi_prev >= rsi_oversold) & (close < bb_lower)
        
        # Exit conditions
        exit_long_condition = (rsi > rsi_overbought + 5) | (rsi < rsi_oversold - 5) | (close < bb_middle)
        exit_short_condition = (rsi < rsi_oversold - 5) | (rsi > rsi_overbought + 5) | (close > bb_middle)
        
        # Generate signals
        long_signals = np.where(long_condition, 1.0, 0.0)
        short_signals = np.where(short_condition, -1.0, 0.0)
        
        # Simple approach to avoid overlapping signals
        # Long signal overrides previous short, vice versa
        position = 0
        for i in range(len(signals)):
            if long_signals[i] == 1.0:
                signals.iloc[i] = 1.0
                position = 1
            elif short_signals[i] == -1.0:
                signals.iloc[i] = -1.0
                position = -1
            elif position == 1 and exit_long_condition[i]:
                signals.iloc[i] = 0.0
                position = 0
            elif position == -1 and exit_short_condition[i]:
                signals.iloc[i] = 0.0
                position = 0
            elif position != 0:
                signals.iloc[i] = position
        
        return signals