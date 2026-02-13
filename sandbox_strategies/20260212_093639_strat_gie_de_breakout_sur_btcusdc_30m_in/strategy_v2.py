from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_30m_keltner_supertrend_breakout")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "supertrend", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"keltner_multiplier": 1.5, "keltner_period": 20, "rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 2.0, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "keltner_multiplier": ParameterSpec(1.0, 3.0, 0.1),
            "keltner_period": ParameterSpec(10, 50, 1),
            "rsi_overbought": ParameterSpec(60, 90, 1),
            "rsi_oversold": ParameterSpec(10, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.1),
            "supertrend_multiplier": ParameterSpec(1.0, 5.0, 0.1),
            "supertrend_period": ParameterSpec(5, 20, 1),
            "tp_atr_mult": ParameterSpec(2.0, 10.0, 0.1),
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
        keltner = indicators["keltner"]
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_lower = np.nan_to_num(keltner["lower"])
        supertrend = indicators["supertrend"]
        supertrend_direction = np.nan_to_num(supertrend["direction"])
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Extract params
        keltner_multiplier = params.get("keltner_multiplier", 1.5)
        keltner_period = params.get("keltner_period", 20)
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_period = params.get("rsi_period", 14)
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        supertrend_multiplier = params.get("supertrend_multiplier", 3.0)
        supertrend_period = params.get("supertrend_period", 10)
        tp_atr_mult = params.get("tp_atr_mult", 5.0)
        warmup = int(params.get("warmup", 50))
        
        # Entry conditions
        # Short entry: price breaks Keltner upper band, supertrend is in downtrend, RSI not overbought
        entry_short = (df["close"].values > keltner_upper) & (supertrend_direction < 0) & (rsi < rsi_overbought)
        
        # Exit conditions
        # Exit on re-entry to Keltner range or trailing stop based on ATR
        exit_short = (df["close"].values < keltner_lower) | (df["close"].values > keltner_upper)
        
        # Initialize signals
        signals.iloc[:warmup] = 0.0
        
        # Generate signals
        entry_mask = entry_short
        exit_mask = exit_short
        
        # Set short signals
        signals[entry_mask] = -1.0
        
        # Set exit signals
        signals[exit_mask] = 0.0
        
        # Ensure only one signal per bar
        # Reset signals to 0.0 where exit conditions are met
        signals = signals.where(~exit_mask, 0.0)
        
        # Set short signal where entry conditions are met
        signals = signals.where(~entry_mask, -1.0)
        
        return signals