from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="keltner_supertrend_breakout_v3")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "supertrend", "atr", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"keltner_multiplier": 1.5, "keltner_period": 20, "rsi_period": 14, "stop_atr_mult": 1.5, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 3.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "keltner_multiplier": ParameterSpec(1.0, 3.0, 0.1),
            "keltner_period": ParameterSpec(10, 50, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "supertrend_multiplier": ParameterSpec(1.0, 5.0, 0.1),
            "supertrend_period": ParameterSpec(5, 20, 1),
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
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Extract indicators
        keltner = indicators["keltner"]
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_lower = np.nan_to_num(keltner["lower"])
        supertrend = indicators["supertrend"]
        supertrend_direction = np.nan_to_num(supertrend["direction"])
        atr = np.nan_to_num(indicators["atr"])
        rsi = np.nan_to_num(indicators["rsi"])
        close = np.nan_to_num(df["close"].values)
        
        # Entry conditions
        entry_condition = (close > keltner_upper) & (supertrend_direction == 1) & (rsi > 50) & (close > np.roll(close, 1))
        
        # Initialize entry price and trailing stop
        entry_price = np.full_like(close, np.nan)
        trailing_stop = np.full_like(close, np.nan)
        
        # Generate signals
        long_positions = np.zeros_like(close, dtype=bool)
        for i in range(1, len(close)):
            if entry_condition[i]:
                entry_price[i] = close[i]
                long_positions[i] = True
            elif long_positions[i-1]:
                # Update trailing stop
                if not np.isnan(entry_price[i-1]):
                    trailing_stop[i] = entry_price[i-1] + params["stop_atr_mult"] * atr[i-1]
                else:
                    trailing_stop[i] = np.nan
                
                # Exit conditions
                exit_condition = (close[i] < keltner_lower[i]) | (supertrend_direction[i] == -1) | (close[i] < trailing_stop[i])
                if exit_condition:
                    long_positions[i] = False
                else:
                    long_positions[i] = True
        
        signals[long_positions] = 1.0
        
        return signals