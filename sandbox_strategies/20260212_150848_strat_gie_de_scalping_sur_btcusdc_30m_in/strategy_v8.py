from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="scalping_bollinger_vwap_atr_rsi")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "vwap", "atr", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 90, 1),
            "rsi_oversold": ParameterSpec(10, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(1.5, 5.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1),
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
        vwap = np.nan_to_num(indicators["vwap"])
        atr = np.nan_to_num(indicators["atr"])
        rsi = np.nan_to_num(indicators["rsi"])
        
        # Extract bollinger bands
        bb_upper = np.nan_to_num(bb["upper"])
        bb_lower = np.nan_to_num(bb["lower"])
        close = np.nan_to_num(df["close"].values)
        
        # Extract parameters
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))
        
        # Initialize position tracking
        position = 0
        entry_price = 0.0
        stop_level = 0.0
        tp_level = 0.0
        
        # Precompute conditions
        long_condition = (close < bb_lower) & (vwap < close) & (rsi < rsi_oversold)
        short_condition = (close > bb_upper) & (vwap > close) & (rsi > rsi_overbought)
        
        # Initialize signals array
        signals_values = np.zeros(len(df))
        
        # Loop through each bar to generate signals
        for i in range(warmup, len(df)):
            if position == 0:
                if long_condition[i]:
                    position = 1
                    entry_price = close[i]
                    stop_level = entry_price - (stop_atr_mult * atr[i])
                    tp_level = entry_price + (tp_atr_mult * atr[i])
                    signals_values[i] = 1.0
                elif short_condition[i]:
                    position = -1
                    entry_price = close[i]
                    stop_level = entry_price + (stop_atr_mult * atr[i])
                    tp_level = entry_price - (tp_atr_mult * atr[i])
                    signals_values[i] = -1.0
            else:
                # Exit conditions
                if position > 0:
                    if close[i] >= tp_level or close[i] <= stop_level:
                        position = 0
                        signals_values[i] = 0.0
                else:  # position < 0
                    if close[i] <= tp_level or close[i] >= stop_level:
                        position = 0
                        signals_values[i] = 0.0
        
        signals.iloc[:] = signals_values
        return signals