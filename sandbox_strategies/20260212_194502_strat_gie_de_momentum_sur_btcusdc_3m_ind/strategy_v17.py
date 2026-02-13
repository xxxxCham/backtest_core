from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="momentum_acceleration_with_atr_filter")

    @property
    def required_indicators(self) -> List[str]:
        return ["macd", "roc", "atr", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(50, 90, 1),
            "rsi_oversold": ParameterSpec(10, 50, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        macd = indicators["macd"]
        roc = np.nan_to_num(indicators["roc"])
        atr = np.nan_to_num(indicators["atr"])
        rsi = np.nan_to_num(indicators["rsi"])
        
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))
        
        signals.iloc[:warmup] = 0.0
        
        # Entry conditions
        long_condition = (
            (np.nan_to_num(macd["macd"]) > np.nan_to_num(macd["signal"])) &
            (roc > 0) &
            (roc > np.roll(roc, 1)) &
            (rsi < rsi_overbought)
        )
        
        short_condition = (
            (np.nan_to_num(macd["macd"]) < np.nan_to_num(macd["signal"])) &
            (roc < 0) &
            (roc < np.roll(roc, 1)) &
            (rsi > rsi_oversold)
        )
        
        # Exit conditions
        exit_long = (
            (np.nan_to_num(macd["macd"]) < np.nan_to_num(macd["signal"])) |
            (roc < 0) |
            (rsi > rsi_overbought) |
            (rsi < rsi_oversold)
        )
        
        exit_short = (
            (np.nan_to_num(macd["macd"]) > np.nan_to_num(macd["signal"])) |
            (roc > 0) |
            (rsi > rsi_overbought) |
            (rsi < rsi_oversold)
        )
        
        # Generate signals
        positions = pd.Series(0.0, index=df.index)
        
        for i in range(1, len(df)):
            if long_condition[i]:
                positions.iloc[i] = 1.0
            elif short_condition[i]:
                positions.iloc[i] = -1.0
            elif positions.iloc[i-1] == 1.0 and exit_long[i]:
                positions.iloc[i] = 0.0
            elif positions.iloc[i-1] == -1.0 and exit_short[i]:
                positions.iloc[i] = 0.0
            else:
                positions.iloc[i] = positions.iloc[i-1]
        
        signals = positions
        return signals