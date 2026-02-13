from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="supertrend_adx_atr_trend_following")

    @property
    def required_indicators(self) -> List[str]:
        return ["supertrend", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"adx_period": 14, "adx_threshold": 25, "atr_threshold": 0.001, "stop_atr_mult": 1.0, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(param_type="int", min_value=5, max_value=30, step=1),
            "adx_threshold": ParameterSpec(param_type="int", min_value=10, max_value=50, step=1),
            "atr_threshold": ParameterSpec(param_type="float", min_value=0.0001, max_value=0.01, step=0.0001),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=0.5, max_value=3.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=5.0, step=0.5),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        supertrend_direction = np.nan_to_num(indicators["supertrend"]["direction"])
        adx_value = np.nan_to_num(indicators["adx"]["adx"])
        atr_value = np.nan_to_num(indicators["atr"])
        
        # Extract params
        adx_threshold = params.get("adx_threshold", 25)
        atr_threshold = params.get("atr_threshold", 0.001)
        stop_atr_mult = params.get("stop_atr_mult", 1.0)
        tp_atr_mult = params.get("tp_atr_mult", 2.0)
        warmup = int(params.get("warmup", 50))
        
        # Entry condition: supertrend direction up, adx above threshold, atr above threshold
        entry_condition = (supertrend_direction > 0) & (adx_value >= adx_threshold) & (atr_value > atr_threshold)
        
        # Exit condition: supertrend direction down, or adx below threshold
        exit_condition = (supertrend_direction < 0) | (adx_value < adx_threshold)
        
        # Generate signals
        positions = np.zeros(len(df))
        in_position = False
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        
        for i in range(len(df)):
            if not in_position and entry_condition[i]:
                positions[i] = 1.0
                in_position = True
                entry_price = df["close"].iloc[i]
                stop_loss = entry_price - (stop_atr_mult * atr_value[i])
                take_profit = entry_price + (tp_atr_mult * atr_value[i])
            elif in_position:
                current_price = df["close"].iloc[i]
                if current_price <= stop_loss or current_price >= take_profit or exit_condition[i]:
                    positions[i] = 0.0
                    in_position = False
                else:
                    positions[i] = 1.0
        
        signals = pd.Series(positions, index=df.index, dtype=np.float64)
        
        # Warmup protection
        signals.iloc[:warmup] = 0.0
        
        return signals