from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mean_reversion_bollinger_stoch_rsi_atri")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "stoch_rsi", "atr", "rsi", "adx"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec("rsi_overbought", 50, 90, 1),
            "rsi_oversold": ParameterSpec("rsi_oversold", 10, 50, 1),
            "rsi_period": ParameterSpec("rsi_period", 5, 30, 1),
            "stop_atr_mult": ParameterSpec("stop_atr_mult", 1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec("tp_atr_mult", 2.0, 5.0, 0.1),
            "warmup": ParameterSpec("warmup", 20, 100, 1),
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

        # Extract indicators
        bb = indicators["bollinger"]
        stoch_rsi_k = np.nan_to_num(indicators["stoch_rsi"]["k"])
        atr = np.nan_to_num(indicators["atr"])
        rsi = np.nan_to_num(indicators["rsi"])
        adx = np.nan_to_num(indicators["adx"]["adx"])

        # Extract Bollinger bands
        lower_bb = np.nan_to_num(bb["lower"])
        middle_bb = np.nan_to_num(bb["middle"])
        upper_bb = np.nan_to_num(bb["upper"])

        # Entry conditions
        entry_long = (np.nan_to_num(df["close"]) < lower_bb) & (stoch_rsi_k < 20) & (rsi < 30) & (adx > 20)

        # Exit condition
        exit_long = np.nan_to_num(df["close"]) > middle_bb

        # Initialize entry and exit points
        entry_points = pd.Series(False, index=df.index)
        exit_points = pd.Series(False, index=df.index)

        # Mark entry and exit points
        entry_points[entry_long] = True
        exit_points[exit_long] = True

        # Generate signals
        long_positions = pd.Series(0.0, index=df.index)
        in_position = False
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0

        for i in range(len(df)):
            if not in_position and entry_points.iloc[i]:
                entry_price = df["close"].iloc[i]
                stop_loss = entry_price - (atr[i] * params["stop_atr_mult"])
                take_profit = entry_price + (atr[i] * params["tp_atr_mult"])
                long_positions.iloc[i] = 1.0
                in_position = True
            elif in_position:
                current_price = df["close"].iloc[i]
                if current_price <= stop_loss or current_price >= take_profit or exit_points.iloc[i]:
                    long_positions.iloc[i] = 0.0
                    in_position = False
                else:
                    long_positions.iloc[i] = 1.0

        signals = long_positions

        return signals