from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btc_bollinger_rsi_scalper")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr", "ema"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(20, 40, 1),
            "rsi_period": ParameterSpec(10, 20, 1),
            "stop_atr_mult": ParameterSpec(1.0, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.5),
            "warmup": ParameterSpec(30, 100, 1),
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

        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        lower = np.nan_to_num(bb["lower"])
        middle = np.nan_to_num(bb["middle"])
        upper = np.nan_to_num(bb["upper"])
        ema_21 = np.nan_to_num(indicators["ema"]["ema_21"])
        atr = np.nan_to_num(indicators["atr"])

        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)

        # Entry long conditions
        price_below_ema_21 = df["close"] < ema_21
        rsi_below_oversold = rsi < rsi_oversold
        rsi_rising = np.roll(rsi, 1) < rsi
        price_above_lower_bb = df["close"] > lower
        rsi_in_oversold = (rsi > rsi_oversold) & (rsi < rsi_oversold + 10)

        long_condition = (
            price_below_ema_21 &
            rsi_below_oversold &
            rsi_rising &
            price_above_lower_bb &
            rsi_in_oversold
        )

        # Entry short conditions
        price_above_ema_21 = df["close"] > ema_21
        rsi_above_overbought = rsi > rsi_overbought
        rsi_falling = np.roll(rsi, 1) > rsi
        price_below_upper_bb = df["close"] < upper
        rsi_in_overbought = (rsi < rsi_overbought) & (rsi > rsi_overbought - 10)

        short_condition = (
            price_above_ema_21 &
            rsi_above_overbought &
            rsi_falling &
            price_below_upper_bb &
            rsi_in_overbought
        )

        # Generate signals
        signals.loc[long_condition] = 1.0
        signals.loc[short_condition] = -1.0

        return signals