from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_stoch_rsi_mean_reversion")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "stoch_rsi", "atr", "volume_oscillator"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "bollinger_period": 20,
            "bollinger_std_dev": 2,
            "stoch_rsi_overbought": 80,
            "stoch_rsi_oversold": 20,
            "stoch_rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "volume_oscillator_long": 10,
            "volume_oscillator_short": 5,
            "warmup": 50
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {}

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        if len(df) <= warmup:
            return signals
        
        # Get indicator arrays
        bb = indicators["bollinger"]
        stoch_rsi = indicators["stoch_rsi"]
        atr_array = indicators["atr"]
        vol_osc = indicators["volume_oscillator"]
        
        # Convert to numpy arrays and handle NaNs
        close = np.nan_to_num(df['close'].values)
        bb_upper = np.nan_to_num(bb["upper"])
        bb_middle = np.nan_to_num(bb["middle"])
        bb_lower = np.nan_to_num(bb["lower"])
        stoch_rsi_k = np.nan_to_num(stoch_rsi["k"])
        atr = np.nan_to_num(atr_array)
        volume_osc = np.nan_to_num(vol_osc)
        
        # Create boolean masks for conditions
        oversold_condition = close < bb_lower
        stoch_rsi_oversold = stoch_rsi_k < params["stoch_rsi_oversold"]
        price_cross_above_lower = close > np.roll(bb_lower, 1)
        volume_condition = volume_osc > 0
        
        # Entry condition (long only)
        entry_long = oversold_condition & stoch_rsi_oversold & price_cross_above_lower & volume_condition
        
        # Exit conditions
        exit_to_middle = close > bb_middle
        stoch_rsi_overbought = stoch_rsi_k > params["stoch_rsi_overbought"]
        exit_condition = exit_to_middle | stoch_rsi_overbought
        
        # Initialize position tracking
        position = 0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        
        for i in range(warmup, len(df)):
            if position == 0 and entry_long[i]:
                position = 1
                entry_price = close[i]
                atr_val = atr[i]
                stop_loss = entry_price - params["stop_atr_mult"] * atr_val
                take_profit = entry_price + params["tp_atr_mult"] * atr_val
                signals.iloc[i] = 1.0
            elif position == 1:
                if exit_condition[i] or close[i] <= stop_loss or close[i] >= take_profit:
                    position = 0
                    signals.iloc[i] = 0.0
                else:
                    signals.iloc[i] = 1.0
            else:
                signals.iloc[i] = 0.0
        
        return signals