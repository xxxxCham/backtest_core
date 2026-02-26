from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="momentum_macd_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["macd", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "leverage": 1,
            "macd_fast": 12,
            "np.nan_to_num(indicators['macd']['signal'])": 9,
            "macd_slow": 26,
            "rsi_period": 14,
            "stop_atr_mult": 2.0,
            "tp_atr_mult": 3.0,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(
                name="rsi_period",
                min_val=5,
                max_val=50,
                default=14,
                param_type="int",
                step=1,
            ),
            "macd_fast": ParameterSpec(
                name="macd_fast",
                min_val=5,
                max_val=30,
                default=12,
                param_type="int",
                step=1,
            ),
            "macd_slow": ParameterSpec(
                name="macd_slow",
                min_val=10,
                max_val=50,
                default=26,
                param_type="int",
                step=1,
            ),
            "np.nan_to_num(indicators['macd']['signal'])": ParameterSpec(
                name="np.nan_to_num(indicators['macd']['signal'])",
                min_val=3,
                max_val=20,
                default=9,
                param_type="int",
                step=1,
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                min_val=0.5,
                max_val=4.0,
                default=2.0,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=1.0,
                max_val=5.0,
                default=3.0,
                param_type="float",
                step=0.1,
            ),
            "leverage": ParameterSpec(
                name="leverage",
                min_val=1,
                max_val=2,
                default=1,
                param_type="int",
                step=1,
            ),
        }

    @staticmethod
    def _cross_up(series_a: np.ndarray, series_b: np.ndarray) -> np.ndarray:
        """Return mask where series_a crosses above series_b."""
        cross = np.zeros_like(series_a, dtype=bool)
        cross[1:] = (series_a[1:] > series_b[1:]) & (series_a[:-1] <= series_b[:-1])
        return cross

    @staticmethod
    def _cross_down(series_a: np.ndarray, series_b: np.ndarray) -> np.ndarray:
        """Return mask where series_a crosses below series_b."""
        cross = np.zeros_like(series_a, dtype=bool)
        cross[1:] = (series_a[1:] < series_b[1:]) & (series_a[:-1] >= series_b[:-1])
        return cross

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get("warmup", 50))

        # Ensure boolean masks are initialized
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        macd = indicators['macd']
        rsi = indicators['rsi']

        long_mask = (
            self._cross_up(indicators['macd']["macd"], indicators['macd']["signal"])
            & (rsi > 35)
            & (rsi < 65)
        )
        short_mask = (
            self._cross_down(indicators['macd']["macd"], indicators['macd']["signal"])
            & (rsi > 35)
            & (rsi < 65)
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals.iloc[:warmup] = 0.0
        return signals