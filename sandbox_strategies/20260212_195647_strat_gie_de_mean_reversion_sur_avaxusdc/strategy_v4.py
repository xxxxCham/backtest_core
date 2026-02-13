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
            "rsi_overbought": ParameterSpec(70, 80, 1),
            "rsi_oversold": ParameterSpec(20, 30, 1),
            "rsi_period": ParameterSpec(10, 20, 1),
            "stop_atr_mult": ParameterSpec(1.0, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 4.0, 0.1),
            "warmup": ParameterSpec(30, 70, 5),
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
        atr = np.nan_to_num(indicators["atr"])
        
        # Extract Bollinger bands
        lower_bb = np.nan_to_num(bb["lower"])
        middle_bb = np.nan_to_num(bb["middle"])
        upper_bb = np.nan_to_num(bb["upper"])
        
        # Extract Stochastic RSI
        stoch_rsi_k = np.nan_to_num(stoch_rsi["k"])
        
        # Entry conditions
        entry_long = (df["close"] < lower_bb) & (stoch_rsi_k < 20)
        
        # Exit condition
        exit_long = df["close"] > middle_bb
        
        # Generate signals
        entry_indices = np.where(entry_long)[0]
        exit_indices = np.where(exit_long)[0]
        
        # Initialize entry and exit tracking
        in_position = False
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        
        # Iterate through entries and exits
        for i in range(len(df)):
            if not in_position and entry_long.iloc[i]:
                entry_price = df["close"].iloc[i]
                stop_loss = entry_price - (atr[i] * params["stop_atr_mult"])
                take_profit = entry_price + (atr[i] * params["tp_atr_mult"])
                signals.iloc[i] = 1.0
                in_position = True
            elif in_position:
                current_price = df["close"].iloc[i]
                if current_price <= stop_loss or current_price >= take_profit or exit_long.iloc[i]:
                    signals.iloc[i] = 0.0
                    in_position = False
                else:
                    signals.iloc[i] = 1.0
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals