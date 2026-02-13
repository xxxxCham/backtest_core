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
        return {"adx_period": 14, "adx_threshold": 25, "sma_fast": 20, "sma_slow": 50, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 100}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(10, 30, 1),
            "adx_threshold": ParameterSpec(10, 40, 1),
            "sma_fast": ParameterSpec(5, 50, 1),
            "sma_slow": ParameterSpec(20, 100, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.5),
            "tp_atr_mult": ParameterSpec(2.0, 10.0, 0.5),
            "warmup": ParameterSpec(50, 200, 10),
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
        sma_fast = int(params.get("sma_fast", 20))
        sma_slow = int(params.get("sma_slow", 50))
        stop_atr_mult = float(params.get("stop_atr_mult", 2.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 5.0))
        warmup = int(params.get("warmup", 100))
        
        sma_fast_vals = np.nan_to_num(indicators["sma"][sma_fast])
        sma_slow_vals = np.nan_to_num(indicators["sma"][sma_slow])
        adx_vals = np.nan_to_num(indicators["adx"]["adx"])
        atr_vals = np.nan_to_num(indicators["atr"])
        
        # Entry short condition: SMA fast crosses below SMA slow with ADX above threshold
        entry_condition = (sma_fast_vals < sma_slow_vals) & (adx_vals > adx_threshold)
        
        # Exit condition: SMA fast crosses above SMA slow OR ADX below threshold
        exit_condition = (sma_fast_vals > sma_slow_vals) | (adx_vals < adx_threshold)
        
        # Initialize position tracking
        position = 0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        
        # Generate signals
        for i in range(warmup, len(signals)):
            if position == 0 and entry_condition[i]:
                position = -1  # Short position
                entry_price = df["close"].iloc[i]
                stop_loss = entry_price + (atr_vals[i] * stop_atr_mult)
                take_profit = entry_price - (atr_vals[i] * tp_atr_mult)
                signals.iloc[i] = -1.0
            elif position == -1:
                # Check exit conditions
                if exit_condition[i]:
                    position = 0
                    signals.iloc[i] = 0.0
                elif df["close"].iloc[i] >= stop_loss:
                    position = 0
                    signals.iloc[i] = 0.0
                elif df["close"].iloc[i] <= take_profit:
                    position = 0
                    signals.iloc[i] = 0.0
            elif position == -1 and df["close"].iloc[i] <= take_profit:
                position = 0
                signals.iloc[i] = 0.0
            elif position == -1 and df["close"].iloc[i] >= stop_loss:
                position = 0
                signals.iloc[i] = 0.0
                
        signals.iloc[:warmup] = 0.0
        return signals