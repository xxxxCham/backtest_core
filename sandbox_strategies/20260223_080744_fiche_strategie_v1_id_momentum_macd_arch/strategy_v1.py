from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="momentum_macd_rsi_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["macd", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "leverage": 1,
            "macd_fast_period": 12,
            "macd_signal_period": 9,
            "macd_slow_period": 26,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.5,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(
                name="rsi_period", min_val=5, max_val=50, default=14, param_type="int", step=1
            ),
            "rsi_oversold": ParameterSpec(
                name="rsi_oversold", min_val=10, max_val=40, default=30, param_type="int", step=1
            ),
            "rsi_overbought": ParameterSpec(
                name="rsi_overbought", min_val=60, max_val=90, default=70, param_type="int", step=1
            ),
            "macd_fast_period": ParameterSpec(
                name="macd_fast_period", min_val=5, max_val=20, default=12, param_type="int", step=1
            ),
            "macd_slow_period": ParameterSpec(
                name="macd_slow_period", min_val=15, max_val=50, default=26, param_type="int", step=1
            ),
            "macd_signal_period": ParameterSpec(
                name="macd_signal_period", min_val=3, max_val=15, default=9, param_type="int", step=1
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult", min_val=0.5, max_val=4.0, default=1.5, param_type="float", step=0.1
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult", min_val=1.0, max_val=5.0, default=3.5, param_type="float", step=0.1
            ),
            "leverage": ParameterSpec(
                name="leverage", min_val=1, max_val=2, default=1, param_type="int", step=1
            ),
        }

    @staticmethod
    def _cross_up(series1: np.ndarray, series2: np.ndarray) -> np.ndarray:
        """Return True where series1 crosses above series2."""
        prev1 = np.roll(series1, 1)
        prev2 = np.roll(series2, 1)
        return (prev1 <= prev2) & (series1 > series2)

    @staticmethod
    def _cross_down(series1: np.ndarray, series2: np.ndarray) -> np.ndarray:
        """Return True where series1 crosses below series2."""
        prev1 = np.roll(series1, 1)
        prev2 = np.roll(series2, 1)
        return (prev1 >= prev2) & (series1 < series2)

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        warmup = int(params.get("warmup", 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        macd = indicators['macd']
        rsi = indicators['rsi']

        macd_cross_up = self._cross_up(indicators['macd']["macd"], indicators['macd']["signal"])
        macd_cross_down = self._cross_down(indicators['macd']["macd"], indicators['macd']["signal"])

        rsi_mask = (rsi > params["rsi_oversold"]) & (rsi < params["rsi_overbought"])

        long_mask = macd_cross_up & rsi_mask
        short_mask = macd_cross_down & rsi_mask

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Zero out initial warmup period
        signals.iloc[:warmup] = 0.0
        return signals