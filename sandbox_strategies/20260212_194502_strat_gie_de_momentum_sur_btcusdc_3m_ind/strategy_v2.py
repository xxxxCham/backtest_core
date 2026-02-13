from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_momentum_three_indicator")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 2.0, "tp_atr_mult": 4.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 90, 1),
            "rsi_oversold": ParameterSpec(10, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 10.0, 0.1),
            "warmup": ParameterSpec(10, 100, 1),
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
        bb = indicators["bollinger"]
        bb_upper = np.nan_to_num(bb["upper"])
        bb_middle = np.nan_to_num(bb["middle"])
        bb_lower = np.nan_to_num(bb["lower"])
        atr = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)
        
        # Parameters
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 4.5)
        warmup = int(params.get("warmup", 50))
        
        # Entry conditions
        entry_long = (rsi < rsi_oversold) & (bb_lower < close) & (close > bb_middle) & (bb_middle > np.roll(bb_middle, 1))
        entry_short = (rsi > rsi_overbought) & (bb_upper > close) & (close < bb_middle) & (bb_middle < np.roll(bb_middle, 1))
        
        # Exit condition
        exit_long = (rsi > rsi_overbought) | (rsi < rsi_oversold)
        exit_short = (rsi > rsi_overbought) | (rsi < rsi_oversold)
        
        # Generate signals
        position = 0
        for i in range(len(df)):
            if i < warmup:
                signals.iloc[i] = 0.0
                continue
                
            if position == 0:
                if entry_long[i]:
                    signals.iloc[i] = 1.0
                    position = 1
                elif entry_short[i]:
                    signals.iloc[i] = -1.0
                    position = -1
            else:
                if (position == 1 and exit_long[i]) or (position == -1 and exit_short[i]):
                    signals.iloc[i] = 0.0
                    position = 0
                else:
                    signals.iloc[i] = position
                    
        return signals