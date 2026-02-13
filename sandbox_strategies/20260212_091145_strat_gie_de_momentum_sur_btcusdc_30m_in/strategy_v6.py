from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="momentum_btcusdc_30m_revised")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(param_type="int", min_value=60, max_value=80, step=5),
            "rsi_oversold": ParameterSpec(param_type="int", min_value=10, max_value=30, step=5),
            "rsi_period": ParameterSpec(param_type="int", min_value=10, max_value=20, step=2),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=2.0, step=0.25),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=4.0, step=0.5),
            "warmup": ParameterSpec(param_type="int", min_value=30, max_value=70, step=10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        bb_upper = np.nan_to_num(bb["upper"])
        bb_lower = np.nan_to_num(bb["lower"])
        atr = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)
        macd = indicators["macd"]
        macd_histogram = np.nan_to_num(macd["histogram"])

        rsi_prev = np.roll(rsi, 1)
        rsi_prev[0] = rsi[0]

        long_condition = (
            (rsi > rsi_overbought) &
            (macd_histogram > 0) &
            (rsi_prev <= rsi_overbought) &
            (close > bb_upper) &
            (rsi > 50)
        )

        short_condition = (
            (rsi < rsi_oversold) &
            (macd_histogram < 0) &
            (rsi_prev >= rsi_oversold) &
            (close < bb_lower) &
            (rsi < 50)
        )

        exit_long_condition = (
            (rsi > rsi_overbought + 5) |
            (rsi < rsi_oversold - 5) |
            ((rsi > 50) & (rsi_prev <= 50))
        )

        exit_short_condition = (
            (rsi < rsi_oversold - 5) |
            (rsi > rsi_overbought + 5) |
            ((rsi < 50) & (rsi_prev >= 50))
        )

        long_signals = np.where(long_condition, 1.0, 0.0)
        short_signals = np.where(short_condition, -1.0, 0.0)

        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        signals.iloc[:warmup] = 0.0

        for i in range(warmup, len(signals)):
            if long_signals[i] == 1.0:
                signals.iloc[i] = 1.0
            elif short_signals[i] == -1.0:
                signals.iloc[i] = -1.0
            elif signals.iloc[i-1] == 1.0 and exit_long_condition[i]:
                signals.iloc[i] = 0.0
            elif signals.iloc[i-1] == -1.0 and exit_short_condition[i]:
                signals.iloc[i] = 0.0

        return signals