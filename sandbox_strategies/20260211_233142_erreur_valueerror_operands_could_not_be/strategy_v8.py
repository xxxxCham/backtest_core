from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_with_risk")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(
                type="int",
                default=14,
                min=7,
                max=28,
                description="Period for RSI calculation."
            ),
            "stop_atr_mult": ParameterSpec(
                type="float",
                default=1.5,
                min=0.5,
                max=3.0,
                description="Multiplier for ATR to calculate stop loss."
            ),
            "tp_atr_mult": ParameterSpec(
                type="float",
                default=3.0,
                min=1.0,
                max=6.0,
                description="Multiplier for ATR to calculate take profit."
            ),
            "warmup": ParameterSpec(
                type="int",
                default=50,
                min=20,
                max=100,
                description="Number of initial bars where no signals are generated."
            )
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        warmup = int(params.get("warmup", 50))
        
        # Apply warmup period
        if warmup > 0:
            signals.iloc[:warmup] = 0.0

        rsi = np.nan_to_num(indicators["rsi"])
        bollinger_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bollinger_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        close = df['close'].values
        atr = np.nan_to_num(indicators["atr"])

        # Calculate stop loss and take profit levels for each position entry
        stop_loss = np.zeros_like(close)
        take_profit = np.zeros_like(close)

        in_position = False
        position_entry_price = 0.0
        current_stop = 0.0
        current_take = 0.0

        for i in range(len(close)):
            if i < warmup:
                signals[i] = 0.0
                continue

            # Long entry conditions
            if close[i] > bollinger_upper[i] and rsi[i] < 30 and not in_position:
                signals[i] = 1.0
                position_entry_price = close[i]
                current_stop = position_entry_price - params["stop_atr_mult"] * atr[i]
                current_take = position_entry_price + (params["tp_atr_mult"] * atr[i])
                in_position = True

            # Short entry conditions
            elif close[i] < bollinger_lower[i] and rsi[i] > 70 and not in_position:
                signals[i] = -1.0
                position_entry_price = close[i]
                current_stop = position_entry_price + params["stop_atr_mult"] * atr[i]
                current_take = position_entry_price - (params["tp_atr_mult"] * atr[i])
                in_position = True

            # Exit conditions if in position
            if in_position:
                if (
                    (signals[i-1] == 1.0 and close[i] <= current_stop) or
                    (signals[i-1] == -1.0 and close[i] >= current_stop)
                ):
                    signals[i] = 0.0
                    in_position = False

                # Exit on RSI returning to normal levels
                elif (
                    (signals[i-1] == 1.0 and rsi[i] > 30) or
                    (signals[i-1] == -1.0 and rsi[i] < 70)
                ):
                    signals[i] = 0.0
                    in_position = False

        return signals