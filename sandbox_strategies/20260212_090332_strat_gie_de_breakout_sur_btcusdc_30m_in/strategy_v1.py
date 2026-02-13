from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="keltner_supertrend_atr_breakout")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "supertrend", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"atr_period": 14, "keltner_multiplier": 1.5, "keltner_period": 20, "stop_atr_mult": 2.0, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "atr_period": ParameterSpec(10, 30, 1),
            "keltner_multiplier": ParameterSpec(1.0, 3.0, 0.1),
            "keltner_period": ParameterSpec(10, 50, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.1),
            "supertrend_multiplier": ParameterSpec(1.0, 5.0, 0.1),
            "supertrend_period": ParameterSpec(5, 20, 1),
            "tp_atr_mult": ParameterSpec(2.0, 10.0, 0.1),
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
        keltner_middle = np.nan_to_num(keltner["middle"])
        
        supertrend = indicators["supertrend"]
        supertrend_line = np.nan_to_num(supertrend["supertrend"])
        supertrend_direction = np.nan_to_num(supertrend["direction"])
        
        atr = np.nan_to_num(indicators["atr"])
        
        # Entry conditions
        price = np.nan_to_num(df["close"].values)
        
        # Long entry: price breaks above Keltner upper band and Supertrend is up
        long_entry = (price > keltner_upper) & (supertrend_direction > 0)
        
        # Short entry: price breaks below Keltner lower band and Supertrend is down
        short_entry = (price < keltner_lower) & (supertrend_direction < 0)
        
        # Exit conditions
        # Exit when price crosses Keltner middle line
        exit_long = price < keltner_middle
        exit_short = price > keltner_middle
        
        # Initialize position tracking
        position = 0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        
        # Track signals
        for i in range(warmup, len(signals)):
            if position == 0:
                if long_entry[i]:
                    position = 1
                    entry_price = price[i]
                    stop_loss = entry_price - (params["stop_atr_mult"] * atr[i])
                    take_profit = entry_price + (params["tp_atr_mult"] * atr[i])
                    signals.iloc[i] = 1.0
                elif short_entry[i]:
                    position = -1
                    entry_price = price[i]
                    stop_loss = entry_price + (params["stop_atr_mult"] * atr[i])
                    take_profit = entry_price - (params["tp_atr_mult"] * atr[i])
                    signals.iloc[i] = -1.0
            else:
                # Check exit conditions
                if position == 1:
                    if exit_long[i] or price[i] <= stop_loss or price[i] >= take_profit:
                        signals.iloc[i] = 0.0
                        position = 0
                elif position == -1:
                    if exit_short[i] or price[i] >= stop_loss or price[i] <= take_profit:
                        signals.iloc[i] = 0.0
                        position = 0
                        
        return signals