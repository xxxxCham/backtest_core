from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_30m_momentum_rsi_bollinger_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

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
            "tp_atr_mult": ParameterSpec(2.0, 6.0, 0.1),
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
        bb_middle = np.nan_to_num(bb["middle"])
        bb_lower = np.nan_to_num(bb["lower"])
        atr = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)
        open_ = np.nan_to_num(df["open"].values)
        
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        warmup = int(params.get("warmup", 50))
        
        signals.iloc[:warmup] = 0.0
        
        # Entry condition: RSI above oversold, rising, price above upper BB, not overbought
        entry_condition = (
            (rsi > rsi_oversold) &
            (rsi > np.roll(rsi, 1)) &
            (close > bb_upper) &
            (rsi < rsi_overbought)
        )
        
        # Exit condition: RSI overbought and falling
        exit_condition = (
            (rsi > rsi_overbought) &
            (rsi < np.roll(rsi, 1))
        )
        
        # Generate signals
        entry_long = entry_condition
        exit_long = exit_condition
        
        # Apply signals
        long_entries = np.where(entry_long, 1.0, 0.0)
        long_exits = np.where(exit_long, -1.0, 0.0)
        
        # Combine entries and exits
        signals = pd.Series(long_entries - long_exits, index=df.index, dtype=np.float64)
        
        # Ensure no overlapping signals
        signal_changes = signals.diff().fillna(0)
        signals = signals.where(signal_changes != 0, 0.0)
        
        return signals