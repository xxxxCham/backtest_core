from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="trend_following_btcusdc_30m_adx_sma_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["sma", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"adx_period": 14, "adx_threshold": 25.0, "sma_period": 50, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 100}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(param_type="int", min_value=5, max_value=30, step=1),
            "adx_threshold": ParameterSpec(param_type="float", min_value=10.0, max_value=50.0, step=1.0),
            "sma_period": ParameterSpec(param_type="int", min_value=20, max_value=100, step=5),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=5.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=10.0, step=0.5),
            "warmup": ParameterSpec(param_type="int", min_value=50, max_value=200, step=10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        adx_period = int(params.get("adx_period", 14))
        adx_threshold = float(params.get("adx_threshold", 25.0))
        sma_period = int(params.get("sma_period", 50))
        stop_atr_mult = float(params.get("stop_atr_mult", 2.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 5.0))
        warmup = int(params.get("warmup", 100))
        
        sma = np.nan_to_num(indicators["sma"])
        adx = np.nan_to_num(indicators["adx"]["adx"])
        atr = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)
        
        # Entry condition: price above SMA and ADX above threshold
        entry_condition = (close > sma) & (adx > adx_threshold)
        
        # Exit condition: SMA below previous SMA and ADX below previous ADX
        sma_shifted = np.roll(sma, 1)
        adx_shifted = np.roll(adx, 1)
        exit_condition = (sma < sma_shifted) & (adx < adx_shifted)
        
        # Generate signals
        entry_signals = pd.Series(0.0, index=df.index)
        entry_signals[entry_condition] = 1.0
        
        exit_signals = pd.Series(0.0, index=df.index)
        exit_signals[exit_condition] = -1.0
        
        # Combine entry and exit signals
        signals = entry_signals + exit_signals
        
        # Set warmup period to flat signals
        signals.iloc[:warmup] = 0.0
        
        return signals