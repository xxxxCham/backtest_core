from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mean_reversion_bollinger_stoch_rsi_atri")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "stoch_rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
            "warmup": ParameterSpec(20, 100, 5)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        bb = indicators["bollinger"]
        stoch_rsi = indicators["stoch_rsi"]
        atr = indicators["atr"]
        
        # Wrap with np.nan_to_num to avoid NaN issues
        close = np.nan_to_num(df["close"].values)
        lower_bb = np.nan_to_num(bb["lower"])
        middle_bb = np.nan_to_num(bb["middle"])
        stoch_rsi_k = np.nan_to_num(stoch_rsi["k"])
        atr_values = np.nan_to_num(atr)
        
        # Entry conditions
        entry_long = (close < lower_bb) & (stoch_rsi_k < 20)
        
        # Exit condition
        exit_long = close > middle_bb
        
        # Generate signals
        long_entries = np.where(entry_long, 1.0, 0.0)
        long_exits = np.where(exit_long, 0.0, 0.0)
        
        # Combine entry and exit signals
        signals_values = np.zeros_like(long_entries)
        in_position = False
        position_entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        
        for i in range(len(signals_values)):
            if not in_position and long_entries[i] == 1.0:
                signals_values[i] = 1.0
                in_position = True
                position_entry_price = close[i]
                stop_loss = position_entry_price - (atr_values[i] * params["stop_atr_mult"])
                take_profit = position_entry_price + (atr_values[i] * params["tp_atr_mult"])
            elif in_position:
                if close[i] <= stop_loss or close[i] >= take_profit or exit_long[i]:
                    signals_values[i] = 0.0
                    in_position = False
                else:
                    signals_values[i] = 1.0
            else:
                signals_values[i] = 0.0
        
        signals = pd.Series(signals_values, index=df.index, dtype=np.float64)
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals