from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="momentum_avaxusdc_1d")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "macd", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(param_type="int", min_value=50, max_value=90, step=5),
            "rsi_oversold": ParameterSpec(param_type="int", min_value=10, max_value=50, step=5),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=2.0, step=0.25),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=4.0, step=0.5),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        rsi = np.nan_to_num(indicators["rsi"])
        macd = indicators["macd"]
        macd_hist = np.nan_to_num(macd["histogram"])
        atr = np.nan_to_num(indicators["atr"])
        
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        
        # Entry conditions
        long_condition = (rsi > rsi_oversold) & (macd_hist > 0) & (macd_hist > np.roll(macd_hist, 1))
        short_condition = (rsi < rsi_overbought) & (macd_hist < 0) & (macd_hist < np.roll(macd_hist, 1))
        
        # Exit conditions
        exit_long_condition = (rsi > rsi_overbought) | (rsi < rsi_oversold) | ((rsi > 50) & (rsi < np.roll(rsi, 1))) | ((rsi < 50) & (rsi > np.roll(rsi, 1)))
        exit_short_condition = (rsi > rsi_overbought) | (rsi < rsi_oversold) | ((rsi > 50) & (rsi < np.roll(rsi, 1))) | ((rsi < 50) & (rsi > np.roll(rsi, 1)))
        
        # Generate signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals