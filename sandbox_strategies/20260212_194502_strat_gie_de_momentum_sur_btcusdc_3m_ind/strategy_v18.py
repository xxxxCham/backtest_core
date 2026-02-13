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
        return ["macd", "roc", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"stop_atr_mult": 2.0, "tp_atr_mult": 4.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                value_type=float,
                min_value=1.0,
                max_value=5.0,
                step=0.5,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                value_type=float,
                min_value=2.0,
                max_value=10.0,
                step=0.5,
            ),
            "warmup": ParameterSpec(
                name="warmup",
                value_type=int,
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
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Extract indicators
        macd = indicators["macd"]
        roc = np.nan_to_num(indicators["roc"])
        atr = np.nan_to_num(indicators["atr"])

        # Prepare arrays for comparison
        macd_macd = np.nan_to_num(macd["macd"])
        macd_signal = np.nan_to_num(macd["signal"])
        roc_lagged = np.roll(roc, 1)

        # Entry conditions
        long_condition = (macd_macd > macd_signal) & (roc > 0) & (roc > roc_lagged)
        short_condition = (macd_macd < macd_signal) & (roc < 0) & (roc < roc_lagged)

        # Exit conditions
        exit_long = (macd_macd < macd_signal) | (roc < 0)
        exit_short = (macd_macd > macd_signal) | (roc > 0)

        # Generate signals
        long_signals = long_condition.astype(int)
        short_signals = short_condition.astype(int)
        exit_long_signals = exit_long.astype(int)
        exit_short_signals = exit_short.astype(int)

        # Create signal series
        long_positions = pd.Series(0.0, index=df.index, dtype=np.float64)
        short_positions = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Initialize positions
        long_positions.iloc[long_condition] = 1.0
        short_positions.iloc[short_condition] = -1.0

        # Handle exits
        long_positions.iloc[exit_long_signals == 1] = 0.0
        short_positions.iloc[exit_short_signals == 1] = 0.0

        # Combine signals
        signals = long_positions + short_positions

        return signals