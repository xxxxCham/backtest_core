from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="keltner_supertrend_breakout")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "supertrend", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"keltner_multiplier": 1.5, "keltner_period": 20, "stop_atr_mult": 1.5, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 3.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "keltner_multiplier": ParameterSpec(1.0, 3.0, 0.1),
            "keltner_period": ParameterSpec(10, 50, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "supertrend_multiplier": ParameterSpec(1.0, 5.0, 0.1),
            "supertrend_period": ParameterSpec(5, 20, 1),
            "tp_atr_mult": ParameterSpec(2.0, 6.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        keltner_mult = params.get("keltner_multiplier", 1.5)
        keltner_period = params.get("keltner_period", 20)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        supertrend_mult = params.get("supertrend_multiplier", 3.0)
        supertrend_period = params.get("supertrend_period", 10)
        tp_atr_mult = params.get("tp_atr_mult", 3.5)
        warmup = int(params.get("warmup", 50))
        
        keltner = indicators["keltner"]
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_lower = np.nan_to_num(keltner["lower"])
        keltner_middle = np.nan_to_num(keltner["middle"])
        
        supertrend = indicators["supertrend"]
        supertrend_trend = np.nan_to_num(supertrend["direction"])
        
        atr = np.nan_to_num(indicators["atr"])
        
        price = np.nan_to_num(df["close"].values)
        
        # Entry conditions
        entry_condition = (price > keltner_upper) & (supertrend_trend == 1)
        
        # Initialize entry points
        entry_points = pd.Series(0.0, index=df.index)
        entry_points[entry_condition] = 1.0
        
        # Generate signals
        long_positions = 0
        for i in range(len(df)):
            if entry_points.iloc[i] == 1.0 and long_positions == 0:
                signals.iloc[i] = 1.0
                long_positions = 1
            elif long_positions == 1:
                # Exit conditions
                if price[i] < keltner_lower[i] or price[i] > keltner_upper[i]:
                    signals.iloc[i] = 0.0
                    long_positions = 0
                else:
                    signals.iloc[i] = 1.0
        
        signals.iloc[:warmup] = 0.0
        return signals