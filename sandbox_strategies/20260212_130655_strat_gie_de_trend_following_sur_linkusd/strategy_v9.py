from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="aroon_sma_trend_following")

    @property
    def required_indicators(self) -> List[str]:
        return ["sma", "aroon", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"aroon_period": 14, "sma_period": 20, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "aroon_period": ParameterSpec(10, 30, 1),
            "sma_period": ParameterSpec(10, 50, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.5),
            "tp_atr_mult": ParameterSpec(3.0, 10.0, 0.5),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        aroon_period = int(params.get("aroon_period", 14))
        sma_period = int(params.get("sma_period", 20))
        stop_atr_mult = float(params.get("stop_atr_mult", 2.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 5.0))
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        sma = np.nan_to_num(indicators["sma"])
        aroon = indicators["aroon"]
        aroon_up = np.nan_to_num(aroon["aroon_up"])
        aroon_down = np.nan_to_num(aroon["aroon_down"])
        atr = np.nan_to_num(indicators["atr"])

        # Calculate SMA slope
        sma_prev = np.roll(sma, 1)
        sma_slope = sma - sma_prev

        # Entry conditions
        entry_long = (sma_slope > 0) & (aroon_up > aroon_down) & (aroon_up > 70)
        entry_short = (sma_slope < 0) & (aroon_down > aroon_up) & (aroon_down > 70)

        # Exit conditions
        exit_long = (sma_slope < 0) | (aroon_down > aroon_up)
        exit_short = (sma_slope > 0) | (aroon_up > aroon_down)

        # Initialize positions
        position = 0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0

        for i in range(warmup, len(df)):
            if position == 0 and entry_long[i]:
                position = 1
                entry_price = df["close"].iloc[i]
                stop_loss = entry_price - (atr[i] * stop_atr_mult)
                take_profit = entry_price + (atr[i] * tp_atr_mult)
                signals.iloc[i] = 1.0
            elif position == 1:
                if exit_long[i] or df["close"].iloc[i] <= stop_loss or df["close"].iloc[i] >= take_profit:
                    position = 0
                    signals.iloc[i] = 0.0
                else:
                    signals.iloc[i] = 1.0
            elif position == 0 and entry_short[i]:
                position = -1
                entry_price = df["close"].iloc[i]
                stop_loss = entry_price + (atr[i] * stop_atr_mult)
                take_profit = entry_price - (atr[i] * tp_atr_mult)
                signals.iloc[i] = -1.0
            elif position == -1:
                if exit_short[i] or df["close"].iloc[i] >= stop_loss or df["close"].iloc[i] <= take_profit:
                    position = 0
                    signals.iloc[i] = 0.0
                else:
                    signals.iloc[i] = -1.0

        return signals