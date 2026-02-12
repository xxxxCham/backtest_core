from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="ema_bollinger_rsi_scalper_v2")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "bollinger", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(param_type="int", min_value=60, max_value=80, step=5),
            "rsi_oversold": ParameterSpec(param_type="int", min_value=10, max_value=30, step=5),
            "rsi_period": ParameterSpec(param_type="int", min_value=10, max_value=20, step=5),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=2.0, step=0.25),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=4.0, step=0.5),
            "warmup": ParameterSpec(param_type="int", min_value=30, max_value=70, step=10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        ema_9 = np.nan_to_num(indicators["ema"])
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_middle = np.nan_to_num(indicators["bollinger"]["middle"])
        bb_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        
        # Price array
        price = np.nan_to_num(df["close"].values)
        
        # RSI parameters
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_period = params.get("rsi_period", 14)
        
        # ATR parameters
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        
        # Warmup period
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Compute previous RSI for divergence
        rsi_prev = np.roll(rsi, 1)
        rsi_prev[0] = rsi[0]
        
        # Entry conditions
        long_condition = (
            (price < bb_middle) &
            (price > bb_lower) &
            (price > ema_9) &
            (rsi < rsi_oversold) &
            (rsi > rsi_prev)
        )
        
        short_condition = (
            (price > bb_middle) &
            (price < bb_upper) &
            (price < ema_9) &
            (rsi > rsi_overbought) &
            (rsi < rsi_prev)
        )
        
        # Exit conditions
        long_exit = (
            (price > bb_upper) |
            ((rsi > rsi_overbought) & (rsi < rsi_prev)) |
            ((rsi < rsi_oversold) & (rsi > rsi_prev))
        )
        
        short_exit = (
            (price < bb_lower) |
            ((rsi > rsi_overbought) & (rsi < rsi_prev)) |
            ((rsi < rsi_oversold) & (rsi > rsi_prev))
        )
        
        # Set signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0
        
        return signals