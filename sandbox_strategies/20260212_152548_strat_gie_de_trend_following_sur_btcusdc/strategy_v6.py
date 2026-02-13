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
        return {"adx_period": 14, "adx_threshold": 25, "sma_fast": 50, "sma_slow": 200, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 250}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(param_name="adx_period", param_type="int", min_value=5, max_value=30, step=5),
            "adx_threshold": ParameterSpec(param_name="adx_threshold", param_type="int", min_value=10, max_value=50, step=5),
            "sma_fast": ParameterSpec(param_name="sma_fast", param_type="int", min_value=20, max_value=100, step=20),
            "sma_slow": ParameterSpec(param_name="sma_slow", param_type="int", min_value=100, max_value=300, step=50),
            "stop_atr_mult": ParameterSpec(param_name="stop_atr_mult", param_type="float", min_value=1.0, max_value=3.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_name="tp_atr_mult", param_type="float", min_value=3.0, max_value=8.0, step=1.0),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Extract indicators
        sma_fast = np.nan_to_num(indicators["sma"])
        sma_slow = np.nan_to_num(indicators["sma"])
        adx = np.nan_to_num(indicators["adx"]["adx"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Extract params
        adx_threshold = params.get("adx_threshold", 25)
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 5.0)
        
        # Entry condition: price below SMA slow and ADX above threshold
        entry_condition = (df["close"] < sma_slow) & (adx > adx_threshold)
        
        # Exit condition: price above SMA slow or ADX below threshold
        exit_condition = (df["close"] > sma_slow) | (adx < adx_threshold)
        
        # Generate signals
        short_entries = entry_condition
        short_exits = exit_condition
        
        # Initialize signals to 0.0 (FLAT)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Set short signals
        signals[short_entries] = -1.0
        signals[short_exits] = 0.0
        
        return signals