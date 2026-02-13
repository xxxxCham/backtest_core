from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_30m_trend_following_adx_sma")

    @property
    def required_indicators(self) -> List[str]:
        return ["sma", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"adx_threshold": 25, "sma_fast": 50, "sma_slow": 200, "stop_atr_mult": 1.0, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_threshold": ParameterSpec(10, 40, 1),
            "sma_fast": ParameterSpec(10, 100, 1),
            "sma_slow": ParameterSpec(100, 300, 1),
            "stop_atr_mult": ParameterSpec(0.5, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec(1.0, 4.0, 0.1),
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

        adx_threshold = params.get("adx_threshold", 25)
        sma_fast = params.get("sma_fast", 50)
        sma_slow = params.get("sma_slow", 200)
        stop_atr_mult = params.get("stop_atr_mult", 1.0)
        tp_atr_mult = params.get("tp_atr_mult", 2.0)

        sma_fast_vals = np.nan_to_num(indicators["sma"][sma_fast])
        sma_slow_vals = np.nan_to_num(indicators["sma"][sma_slow])
        adx_vals = np.nan_to_num(indicators["adx"]["adx"])
        atr_vals = np.nan_to_num(indicators["atr"])

        # Entry short condition: SMA fast crosses below SMA slow, and ADX > threshold
        entry_short = (sma_fast_vals < sma_slow_vals) & (adx_vals > adx_threshold)

        # Exit condition: SMA fast crosses above SMA slow, or ADX < threshold
        exit_short = (sma_fast_vals > sma_slow_vals) | (adx_vals < adx_threshold)

        # Generate signals
        short_positions = np.zeros_like(entry_short, dtype=bool)
        in_short = False

        for i in range(len(entry_short)):
            if not in_short and entry_short[i]:
                in_short = True
                short_positions[i] = True
            elif in_short and exit_short[i]:
                in_short = False
            elif in_short:
                short_positions[i] = True

        signals[short_positions] = -1.0

        return signals