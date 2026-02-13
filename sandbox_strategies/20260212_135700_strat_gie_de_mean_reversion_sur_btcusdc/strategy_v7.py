from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="keltner_cci_mean_reversion_v5")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "cci", "atr", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"cci_period": 14, "keltner_multiplier": 1.5, "keltner_period": 20, "rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "cci_period": ParameterSpec(10, 30, 1),
            "keltner_multiplier": ParameterSpec(1.0, 3.0, 0.1),
            "keltner_period": ParameterSpec(10, 50, 1),
            "rsi_overbought": ParameterSpec(60, 90, 1),
            "rsi_oversold": ParameterSpec(10, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 10.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1)
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
        cci = indicators["cci"]
        atr = indicators["atr"]
        rsi = indicators["rsi"]
        
        # Keltner channel components
        keltner_upper = keltner["upper"]
        keltner_lower = keltner["lower"]
        keltner_middle = keltner["middle"]
        
        # RSI components
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_period = params.get("rsi_period", 14)
        
        # CCI components
        cci_period = params.get("cci_period", 14)
        
        # ATR components
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 5.0)
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Lagged values
        cci_lag1 = np.roll(cci, 1)
        rsi_lag1 = np.roll(rsi, 1)
        
        # Keltner middle slope
        keltner_middle_slope = np.gradient(keltner_middle)
        
        # Entry conditions for long
        entry_long = (
            (df["close"] <= keltner_lower) &
            (cci < -100) &
            (cci_lag1 > -100) &
            (rsi < rsi_oversold) &
            (rsi_lag1 > rsi_oversold) &
            (keltner_middle_slope < 0)
        )
        
        # Exit conditions for long
        exit_long = (
            (df["close"] >= keltner_middle) &
            (rsi > rsi_overbought) &
            (rsi_lag1 < rsi_overbought)
        )
        
        # Generate signals
        long_entries = entry_long & ~np.roll(entry_long, 1)
        long_exits = exit_long & ~np.roll(exit_long, 1)
        
        # Set signals
        signals[long_entries] = 1.0
        signals[long_exits] = 0.0
        
        return signals