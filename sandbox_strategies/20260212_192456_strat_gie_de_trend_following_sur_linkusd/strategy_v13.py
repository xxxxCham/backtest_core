from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="supertrend_adx_trend_following")

    @property
    def required_indicators(self) -> List[str]:
        return ["supertrend", "adx", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"adx_period": 14, "adx_threshold": 25.0, "rsi_period": 14, "stop_atr_mult": 1.0, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(param_name="adx_period", param_type="int", min_value=5, max_value=30, step=1),
            "adx_threshold": ParameterSpec(param_name="adx_threshold", param_type="float", min_value=10.0, max_value=50.0, step=1.0),
            "rsi_period": ParameterSpec(param_name="rsi_period", param_type="int", min_value=5, max_value=30, step=1),
            "stop_atr_mult": ParameterSpec(param_name="stop_atr_mult", param_type="float", min_value=0.5, max_value=3.0, step=0.5),
            "supertrend_multiplier": ParameterSpec(param_name="supertrend_multiplier", param_type="float", min_value=1.0, max_value=5.0, step=0.5),
            "supertrend_period": ParameterSpec(param_name="supertrend_period", param_type="int", min_value=5, max_value=30, step=1),
            "tp_atr_mult": ParameterSpec(param_name="tp_atr_mult", param_type="float", min_value=1.0, max_value=5.0, step=0.5),
            "warmup": ParameterSpec(param_name="warmup", param_type="int", min_value=20, max_value=100, step=10),
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
        supertrend_direction = np.nan_to_num(indicators["supertrend"]["direction"])
        adx_value = np.nan_to_num(indicators["adx"]["adx"])
        rsi_value = np.nan_to_num(indicators["rsi"])
        atr_value = np.nan_to_num(indicators["atr"])
        
        # Entry conditions
        entry_long = (supertrend_direction > 0) & (adx_value >= params["adx_threshold"]) & (rsi_value < 50)
        
        # Exit conditions
        exit_long = (supertrend_direction < 0) | (adx_value < 20.0) | (rsi_value > 50)
        
        # Generate signals
        long_entries = entry_long & ~np.roll(entry_long, 1)
        long_exits = exit_long & ~np.roll(exit_long, 1)
        
        signals.loc[long_entries] = 1.0
        signals.loc[long_exits] = 0.0
        
        return signals