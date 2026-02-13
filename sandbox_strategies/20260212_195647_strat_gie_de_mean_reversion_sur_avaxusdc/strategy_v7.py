from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mean_reversion_bollinger_stoch_rsi_atr")

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
            "rsi_oversold": ParameterSpec(20, 40, 1),
            "rsi_period": ParameterSpec(10, 20, 1),
            "stop_atr_mult": ParameterSpec(1.0, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 4.0, 0.1),
            "warmup": ParameterSpec(30, 70, 1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Extract indicators
        bb = indicators["bollinger"]
        stoch_rsi = indicators["stoch_rsi"]
        atr = indicators["atr"]
        
        # Prepare arrays with nan_to_num
        close = np.nan_to_num(df["close"].values)
        bb_lower = np.nan_to_num(bb["lower"])
        bb_middle = np.nan_to_num(bb["middle"])
        stoch_rsi_k = np.nan_to_num(stoch_rsi["k"])
        atr_values = np.nan_to_num(atr)
        
        # Entry condition: close below lower Bollinger band and Stochastic RSI < 20
        entry_condition = (close < bb_lower) & (stoch_rsi_k < 20)
        
        # Exit condition: close above middle Bollinger band
        exit_condition = close > bb_middle
        
        # Initialize entry and exit signals
        entry_signal = pd.Series(0.0, index=df.index, dtype=np.float64)
        exit_signal = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Set entry signals
        entry_signal[entry_condition] = 1.0
        
        # Set exit signals
        exit_signal[exit_condition] = -1.0
        
        # Combine signals: only long positions allowed
        # Entry and exit are separate signals; we track position
        position = 0
        for i in range(len(signals)):
            if entry_signal.iloc[i] == 1.0:
                position = 1
            elif exit_signal.iloc[i] == -1.0 and position == 1:
                position = 0
            signals.iloc[i] = position * 1.0
            
        return signals