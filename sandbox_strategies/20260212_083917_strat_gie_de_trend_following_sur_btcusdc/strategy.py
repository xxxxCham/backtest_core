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
        return {"adx_period": 14, "atr_period": 14, "sma_period": 20, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(param_type="int", min_value=5, max_value=30, step=1),
            "atr_period": ParameterSpec(param_type="int", min_value=5, max_value=30, step=1),
            "sma_period": ParameterSpec(param_type="int", min_value=5, max_value=50, step=1),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=5.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=10.0, step=0.5),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        adx_period = int(params.get("adx_period", 14))
        atr_period = int(params.get("atr_period", 14))
        sma_period = int(params.get("sma_period", 20))
        stop_atr_mult = float(params.get("stop_atr_mult", 2.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 5.0))
        warmup = int(params.get("warmup", 50))
        
        sma_close = np.nan_to_num(indicators["sma"])
        sma_sma = np.nan_to_num(indicators["sma"])
        adx_adx = np.nan_to_num(indicators["adx"]["adx"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Entry condition: short signal when SMA crosses below SMA and ADX > 25
        entry_condition = (sma_close < sma_sma) & (adx_adx > 25)
        
        # Exit condition: when SMA crosses above SMA or ADX < 15
        exit_condition = (sma_close > sma_sma) | (adx_adx < 15)
        
        # Generate short signals
        short_signal = np.zeros_like(entry_condition, dtype=np.float64)
        short_signal[entry_condition] = -1.0
        
        # Apply exit condition to close existing shorts
        in_short_position = False
        for i in range(len(short_signal)):
            if short_signal[i] == -1.0:
                in_short_position = True
            elif in_short_position and exit_condition[i]:
                short_signal[i] = 0.0
                in_short_position = False
            elif in_short_position:
                short_signal[i] = -1.0
        
        signals = pd.Series(short_signal, index=df.index, dtype=np.float64)
        signals.iloc[:warmup] = 0.0
        return signals