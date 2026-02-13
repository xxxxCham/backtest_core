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
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(param_type="int", min_value=50, max_value=90, step=5),
            "rsi_oversold": ParameterSpec(param_type="int", min_value=10, max_value=50, step=5),
            "rsi_period": ParameterSpec(param_type="int", min_value=5, max_value=30, step=5),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=3.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=5.0, step=0.5),
            "warmup": ParameterSpec(param_type="int", min_value=30, max_value=100, step=10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        rsi = np.nan_to_num(indicators["rsi"])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        close = np.nan_to_num(df["close"].values)
        atr = np.nan_to_num(indicators["atr"])
        
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        warmup = int(params.get("warmup", 50))
        
        signals.iloc[:warmup] = 0.0
        
        # Entry conditions
        rsi_long_condition = (rsi > rsi_oversold) & (np.roll(rsi, 1) <= rsi_oversold)
        price_long_condition = (close > bb_upper) & (close > np.roll(close, 1))
        long_entry = rsi_long_condition & price_long_condition
        
        rsi_short_condition = (rsi < rsi_overbought) & (np.roll(rsi, 1) >= rsi_overbought)
        price_short_condition = (close < bb_lower) & (close < np.roll(close, 1))
        short_entry = rsi_short_condition & price_short_condition
        
        # Exit conditions
        exit_long = (rsi > rsi_overbought) | (rsi < rsi_oversold) | (close > bb_upper) | (close < bb_lower)
        exit_short = (rsi > rsi_overbought) | (rsi < rsi_oversold) | (close > bb_upper) | (close < bb_lower)
        
        # Generate signals
        long_signals = np.zeros_like(rsi)
        short_signals = np.zeros_like(rsi)
        
        long_signals[long_entry] = 1.0
        short_signals[short_entry] = -1.0
        
        # Apply exits
        for i in range(1, len(rsi)):
            if long_signals[i-1] == 1.0:
                if exit_long[i]:
                    long_signals[i] = 0.0
                else:
                    long_signals[i] = 1.0
            elif short_signals[i-1] == -1.0:
                if exit_short[i]:
                    short_signals[i] = 0.0
                else:
                    short_signals[i] = -1.0
        
        signals = pd.Series(long_signals - short_signals, index=df.index, dtype=np.float64)
        
        return signals