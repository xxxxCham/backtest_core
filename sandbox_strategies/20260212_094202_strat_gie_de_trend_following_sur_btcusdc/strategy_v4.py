from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btc_sma_adx_atr_short")

    @property
    def required_indicators(self) -> List[str]:
        return ["sma", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"adx_period": 14, "adx_threshold": 25, "sma_period": 20, "stop_atr_mult": 1.0, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(param_name="adx_period", param_type="int", min_value=5, max_value=30, step=1),
            "adx_threshold": ParameterSpec(param_name="adx_threshold", param_type="int", min_value=10, max_value=50, step=5),
            "sma_period": ParameterSpec(param_name="sma_period", param_type="int", min_value=10, max_value=50, step=5),
            "stop_atr_mult": ParameterSpec(param_name="stop_atr_mult", param_type="float", min_value=0.5, max_value=2.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_name="tp_atr_mult", param_type="float", min_value=1.0, max_value=4.0, step=0.5),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        adx_period = int(params.get("adx_period", 14))
        adx_threshold = float(params.get("adx_threshold", 25))
        sma_period = int(params.get("sma_period", 20))
        stop_atr_mult = float(params.get("stop_atr_mult", 1.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.0))
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        sma = np.nan_to_num(indicators["sma"])
        adx = np.nan_to_num(indicators["adx"]["adx"])
        atr = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)
        
        # Entry condition: price below SMA and ADX above threshold
        entry_condition = (close < sma) & (adx > adx_threshold)
        
        # Exit condition: divergence - SMA rising while ADX falling
        sma_shifted = np.roll(sma, 1)
        adx_shifted = np.roll(adx, 1)
        exit_condition = (sma > sma_shifted) & (adx < adx_shifted)
        
        # Short signal
        short_signal = entry_condition & ~exit_condition
        
        # Set signals
        signals[short_signal] = -1.0
        
        return signals