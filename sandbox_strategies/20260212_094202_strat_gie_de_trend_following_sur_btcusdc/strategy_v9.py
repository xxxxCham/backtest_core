from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_30m_trend_following_adx_sma_revised")

    @property
    def required_indicators(self) -> List[str]:
        return ["sma", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"adx_period": 14, "sma_fast": 50, "sma_slow": 200, "stop_atr_mult": 1.0, "tp_atr_mult": 2.0, "warmup": 250}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec("adx_period", 10, 30, 14),
            "sma_fast": ParameterSpec("sma_fast", 20, 100, 50),
            "sma_slow": ParameterSpec("sma_slow", 150, 300, 200),
            "stop_atr_mult": ParameterSpec("stop_atr_mult", 0.5, 2.0, 1.0),
            "tp_atr_mult": ParameterSpec("tp_atr_mult", 1.0, 4.0, 2.0),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        adx_period = int(params.get("adx_period", 14))
        sma_fast = int(params.get("sma_fast", 50))
        sma_slow = int(params.get("sma_slow", 200))
        stop_atr_mult = float(params.get("stop_atr_mult", 1.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.0))
        warmup = int(params.get("warmup", 250))
        signals.iloc[:warmup] = 0.0
        sma_fast_vals = np.nan_to_num(indicators["sma"][sma_fast])
        sma_slow_vals = np.nan_to_num(indicators["sma"][sma_slow])
        adx_vals = np.nan_to_num(indicators["adx"]["adx"])
        atr_vals = np.nan_to_num(indicators["atr"])
        short_condition = (sma_fast_vals < sma_slow_vals) & (adx_vals > 25)
        exit_condition = (sma_fast_vals > sma_slow_vals) | (adx_vals < 20)
        short_signals = pd.Series(0.0, index=df.index)
        short_signals[short_condition] = -1.0
        short_signals[exit_condition] = 0.0
        signals = short_signals
        return signals