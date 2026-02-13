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
        return {"stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                param_type="float",
                min_value=1.0,
                max_value=3.0,
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                param_type="float",
                min_value=2.0,
                max_value=6.0,
                step=0.1,
            ),
            "warmup": ParameterSpec(
                name="warmup",
                param_type="int",
                min_value=20,
                max_value=100,
                step=10,
            ),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        macd = indicators["macd"]
        roc = np.nan_to_num(indicators["roc"])
        atr = np.nan_to_num(indicators["atr"])
        rsi = np.nan_to_num(indicators["rsi"])
        
        # Entry conditions
        long_condition = (
            (np.nan_to_num(macd["macd"]) > np.nan_to_num(macd["signal"])) &
            (roc > 0) &
            (roc > np.roll(roc, 1)) &
            (rsi < 50)
        )
        
        short_condition = (
            (np.nan_to_num(macd["macd"]) < np.nan_to_num(macd["signal"])) &
            (roc < 0) &
            (roc < np.roll(roc, 1)) &
            (rsi > 50)
        )
        
        # Exit conditions
        exit_long = (
            (np.nan_to_num(macd["macd"]) < np.nan_to_num(macd["signal"])) |
            (roc < 0)
        )
        
        exit_short = (
            (np.nan_to_num(macd["macd"]) > np.nan_to_num(macd["signal"])) |
            (roc > 0)
        )
        
        # Initialize position tracking
        position = 0
        position_change = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        for i in range(warmup, len(df)):
            if position == 0:
                if long_condition[i]:
                    position = 1
                    position_change.iloc[i] = 1.0
                elif short_condition[i]:
                    position = -1
                    position_change.iloc[i] = -1.0
            elif position == 1:
                if exit_long[i]:
                    position = 0
                    position_change.iloc[i] = 0.0
            elif position == -1:
                if exit_short[i]:
                    position = 0
                    position_change.iloc[i] = 0.0
        
        signals.iloc[warmup:] = position_change.iloc[warmup:]
        return signals