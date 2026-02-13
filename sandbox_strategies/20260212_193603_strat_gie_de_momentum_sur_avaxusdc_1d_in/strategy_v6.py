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
            "rsi_overbought": ParameterSpec(50, 90, 1),
            "rsi_oversold": ParameterSpec(10, 50, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        bb_upper = np.nan_to_num(bb["upper"])
        bb_lower = np.nan_to_num(bb["lower"])
        atr = np.nan_to_num(indicators["atr"])
        price = np.nan_to_num(df["close"].values)
        
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        warmup = int(params.get("warmup", 50))
        
        # Entry conditions
        rsi_long_condition = (rsi > rsi_oversold) & (rsi > np.roll(rsi, 1))
        rsi_short_condition = (rsi < rsi_overbought) & (rsi < np.roll(rsi, 1))
        
        # Confirm with price action
        price_long_condition = price > bb_upper
        price_short_condition = price < bb_lower
        
        # MACD confirmation (using MACD histogram)
        macd_hist = np.nan_to_num(indicators["macd"]["histogram"]) if "macd" in indicators else np.zeros_like(rsi)
        macd_long_condition = macd_hist > 0
        macd_short_condition = macd_hist < 0
        
        # Combine all conditions
        long_entry = rsi_long_condition & price_long_condition & macd_long_condition
        short_entry = rsi_short_condition & price_short_condition & macd_short_condition
        
        # Exit conditions
        exit_long_condition = (rsi > rsi_overbought) | (rsi < np.roll(rsi, 1))
        exit_short_condition = (rsi < rsi_oversold) | (rsi > np.roll(rsi, 1))
        
        # Initialize signal array
        signal_array = np.zeros_like(rsi)
        
        # Generate signals
        long_positions = np.zeros_like(rsi, dtype=bool)
        short_positions = np.zeros_like(rsi, dtype=bool)
        
        for i in range(1, len(rsi)):
            if long_entry[i] and not long_positions[i-1] and not short_positions[i-1]:
                signal_array[i] = 1.0
                long_positions[i] = True
            elif short_entry[i] and not long_positions[i-1] and not short_positions[i-1]:
                signal_array[i] = -1.0
                short_positions[i] = True
            elif long_positions[i-1] and exit_long_condition[i]:
                signal_array[i] = 0.0
                long_positions[i] = False
            elif short_positions[i-1] and exit_short_condition[i]:
                signal_array[i] = 0.0
                short_positions[i] = False
        
        signals = pd.Series(signal_array, index=df.index, dtype=np.float64)
        signals.iloc[:warmup] = 0.0
        
        return signals