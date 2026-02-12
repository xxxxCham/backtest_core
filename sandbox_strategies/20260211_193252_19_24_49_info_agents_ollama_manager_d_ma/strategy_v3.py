from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="ema_bollinger_rsi_scalper")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "bollinger", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(20, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
            "warmup": ParameterSpec(20, 100, 5),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract and sanitize indicators
        ema_9 = np.nan_to_num(indicators["ema"])
        ema_21 = np.nan_to_num(indicators["ema"])
        ema_50 = np.nan_to_num(indicators["ema"])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_middle = np.nan_to_num(indicators["bollinger"]["middle"])
        bb_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Use specific EMA values (assuming ema array is ordered by period)
        # For this strategy, we'll use ema_9, ema_21, ema_50 from the ema array
        # We'll assume ema array is sorted with 9, 21, 50 period EMA in that order
        ema_9 = ema_9[0] if isinstance(ema_9, np.ndarray) and len(ema_9) > 0 else np.full_like(rsi, 0)
        ema_21 = ema_21[1] if isinstance(ema_21, np.ndarray) and len(ema_21) > 1 else np.full_like(rsi, 0)
        ema_50 = ema_50[2] if isinstance(ema_50, np.ndarray) and len(ema_50) > 2 else np.full_like(rsi, 0)
        
        # RSI parameters
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        
        # Stop loss and take profit multipliers
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        
        # Warmup
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Create condition arrays
        price = df["close"].values
        rsi_prev = np.roll(rsi, 1)
        rsi_prev[0] = rsi[0]
        
        # Long entry conditions
        long_condition_1 = price > ema_21
        long_condition_2 = price > ema_9
        long_condition_3 = rsi < rsi_oversold
        long_condition_4 = rsi > rsi_prev
        long_condition_5 = price > bb_lower
        long_condition_6 = price < bb_middle
        
        long_entry = long_condition_1 & long_condition_2 & long_condition_3 & long_condition_4 & long_condition_5 & long_condition_6
        
        # Short entry conditions
        short_condition_1 = price < ema_21
        short_condition_2 = price < ema_9
        short_condition_3 = rsi > rsi_overbought
        short_condition_4 = rsi < rsi_prev
        short_condition_5 = price < bb_upper
        short_condition_6 = price > bb_middle
        
        short_entry = short_condition_1 & short_condition_2 & short_condition_3 & short_condition_4 & short_condition_5 & short_condition_6
        
        # Exit conditions
        exit_long = (price > bb_upper) | (price < bb_lower) | ((rsi > 70) & (rsi < rsi_prev)) | ((rsi < 30) & (rsi > rsi_prev))
        exit_short = (price < bb_lower) | (price > bb_upper) | ((rsi > 70) & (rsi < rsi_prev)) | ((rsi < 30) & (rsi > rsi_prev))
        
        # Generate signals
        long_signals = np.where(long_entry, 1.0, 0.0)
        short_signals = np.where(short_entry, -1.0, 0.0)
        
        # Combine signals
        signals = pd.Series(long_signals + short_signals, index=df.index, dtype=np.float64)
        
        # Ensure no conflicting signals at the same time
        # If both long and short signals are active, prefer long (or use a rule to resolve)
        # In this case, we'll just keep the first valid signal, so we'll apply the signals in order
        
        # Apply warmup protection
        signals.iloc[:warmup] = 0.0
        
        return signals