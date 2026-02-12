from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_atr_strategy")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"bollinger_period": 20, "bollinger_std_dev": 2, "rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 2.0, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            # fill each tunable parameter
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

        # Access the indicators
        bollinger = indicators["bollinger"]
        rsi = np.nan_to_num(indicators["rsi"])

        # Parameters
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)

        # Entry conditions
        long_entry = (np.nan_to_num(df["close"]) > bollinger["upper"]) & (rsi < rsi_oversold)
        short_entry = (np.nan_to_num(df["close"]) < bollinger["lower"]) & (rsi > rsi_overbought)

        # Exit conditions
        exit_condition = (np.nan_to_num(df["close"]) == np.nan_to_num(bollinger["middle"])) | (rsi == 50.0)

        # Apply signals with delays to avoid overlapping
        entries_long = long_entry & ~signals.shift(1).isin([1.0, -1.0])  # Avoid holding both positions
        entries_short = short_entry & ~signals.shift(1).isin([1.0, -1.0])

        # Set signals
        signals[entries_long] = 1.0
        signals[entries_short] = -1.0

        # Exit existing positions when exit condition is met
        current_positions = signals.shift(1)
        exits = (current_positions != 0) & exit_condition
        signals[exits] = 0.0  # Flatten position

        return signals