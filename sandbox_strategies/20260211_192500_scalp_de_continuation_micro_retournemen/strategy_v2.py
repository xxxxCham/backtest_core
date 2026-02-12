from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btc_ema_rsi_bollinger_scalper")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi", "bollinger", "atr"]

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
            "tp_atr_mult": ParameterSpec(2.0, 4.0, 0.1),
            "warmup": ParameterSpec(30, 70, 1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_period = params.get("rsi_period", 14)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        ema_9 = np.nan_to_num(indicators["ema"]["ema_9"])
        ema_21 = np.nan_to_num(indicators["ema"]["ema_21"])
        ema_50 = np.nan_to_num(indicators["ema"]["ema_50"])
        rsi = np.nan_to_num(indicators["rsi"])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_middle = np.nan_to_num(indicators["bollinger"]["middle"])
        bb_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        atr = np.nan_to_num(indicators["atr"])

        price = np.nan_to_num(df["close"].values)
        prev_price = np.roll(price, 1)
        prev_rsi = np.roll(rsi, 1)

        # Entry long conditions
        long_condition = (
            (price > ema_21) &
            (prev_price <= ema_21) &
            (rsi > rsi_oversold) &
            (rsi > prev_rsi) &
            (price < bb_upper) &
            (price > bb_lower) &
            (prev_price <= bb_lower) &
            (price > prev_price)
        )

        # Entry short conditions
        short_condition = (
            (price < ema_21) &
            (prev_price >= ema_21) &
            (rsi < rsi_overbought) &
            (rsi < prev_rsi) &
            (price > bb_lower) &
            (price < bb_upper) &
            (prev_price >= bb_upper) &
            (price < prev_price)
        )

        # Generate signals
        long_signals = np.where(long_condition, 1.0, 0.0)
        short_signals = np.where(short_condition, -1.0, 0.0)

        signals = pd.Series(long_signals + short_signals, index=df.index, dtype=np.float64)

        return signals