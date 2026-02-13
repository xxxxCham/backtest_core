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
        return {"adx_period": 14, "adx_threshold": 20, "sma_fast": 50, "sma_slow": 200, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 250}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(param_type="int", min_value=5, max_value=30, step=1),
            "adx_threshold": ParameterSpec(param_type="float", min_value=10.0, max_value=50.0, step=5.0),
            "sma_fast": ParameterSpec(param_type="int", min_value=20, max_value=100, step=10),
            "sma_slow": ParameterSpec(param_type="int", min_value=100, max_value=300, step=20),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=5.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=3.0, max_value=10.0, step=1.0),
            "warmup": ParameterSpec(param_type="int", min_value=100, max_value=500, step=50),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        adx_period = int(params.get("adx_period", 14))
        adx_threshold = float(params.get("adx_threshold", 20))
        sma_fast = int(params.get("sma_fast", 50))
        sma_slow = int(params.get("sma_slow", 200))
        stop_atr_mult = float(params.get("stop_atr_mult", 2.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 5.0))
        warmup = int(params.get("warmup", 250))
        signals.iloc[:warmup] = 0.0

        sma_fast_vals = np.nan_to_num(indicators["sma"][sma_fast])
        sma_slow_vals = np.nan_to_num(indicators["sma"][sma_slow])
        adx_vals = np.nan_to_num(indicators["adx"]["adx"])
        atr_vals = np.nan_to_num(indicators["atr"])

        # Entry condition: SMA fast crosses below SMA slow AND ADX > threshold
        entry_condition = (sma_fast_vals < sma_slow_vals) & (np.roll(sma_fast_vals, 1) >= np.roll(sma_slow_vals, 1))

        # Trend filter: ADX must be above threshold
        trend_filter = adx_vals > adx_threshold

        # Exit condition: ADX below threshold OR SMA fast crosses above SMA slow
        exit_condition = (adx_vals < adx_threshold) | (sma_fast_vals > sma_slow_vals) & (np.roll(sma_fast_vals, 1) <= np.roll(sma_slow_vals, 1))

        # Generate short signals
        short_entry = entry_condition & trend_filter
        short_exit = exit_condition

        # Convert boolean masks to signals
        short_signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        short_signals.loc[short_entry] = -1.0
        short_signals.loc[short_exit] = 0.0

        # Apply signals
        signals.loc[short_signals == -1.0] = -1.0
        signals.loc[short_signals == 0.0] = 0.0

        return signals