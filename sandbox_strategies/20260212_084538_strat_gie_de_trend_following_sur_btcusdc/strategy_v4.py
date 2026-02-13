from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_trend_following_sma_adx_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["sma", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"adx_threshold": 25, "sma_fast": 10, "sma_slow": 30, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_threshold": ParameterSpec(10, 50, 1),
            "sma_fast": ParameterSpec(5, 20, 1),
            "sma_slow": ParameterSpec(20, 60, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.5),
            "tp_atr_mult": ParameterSpec(2.0, 10.0, 0.5),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        adx_threshold = params.get("adx_threshold", 25)
        sma_fast = params.get("sma_fast", 10)
        sma_slow = params.get("sma_slow", 30)
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 5.0)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        sma_fast_vals = np.nan_to_num(indicators["sma"][sma_fast])
        sma_slow_vals = np.nan_to_num(indicators["sma"][sma_slow])
        adx_vals = np.nan_to_num(indicators["adx"]["adx"])
        atr_vals = np.nan_to_num(indicators["atr"])
        
        # Entry long: SMA fast crosses above SMA slow with ADX > threshold
        entry_long = (sma_fast_vals > sma_slow_vals) & (adx_vals > adx_threshold)
        
        # Exit long: SMA fast crosses below SMA slow with ADX < threshold
        exit_long = (sma_fast_vals < sma_slow_vals) & (adx_vals < adx_threshold)
        
        # Create signal array
        entry_mask = entry_long
        exit_mask = exit_long
        
        # Initialize signal array
        signal_array = np.zeros(len(df))
        
        # Track position
        in_position = False
        entry_index = -1
        
        for i in range(len(df)):
            if not in_position and entry_mask[i]:
                signal_array[i] = 1.0  # Enter long
                in_position = True
                entry_index = i
            elif in_position and exit_mask[i]:
                signal_array[i] = -1.0  # Exit long
                in_position = False
                entry_index = -1
            elif in_position:
                # Check for stop loss or take profit
                if entry_index >= 0:
                    stop_loss = df["close"].iloc[entry_index] - stop_atr_mult * atr_vals[entry_index]
                    take_profit = df["close"].iloc[entry_index] + tp_atr_mult * atr_vals[entry_index]
                    current_price = df["close"].iloc[i]
                    if current_price <= stop_loss or current_price >= take_profit:
                        signal_array[i] = -1.0  # Exit due to SL or TP
                        in_position = False
                        entry_index = -1
        
        signals = pd.Series(signal_array, index=df.index, dtype=np.float64)
        return signals