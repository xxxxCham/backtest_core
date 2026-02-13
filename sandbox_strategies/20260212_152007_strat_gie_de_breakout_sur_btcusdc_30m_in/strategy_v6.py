from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_keltner_supertrend_breakout_short")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "supertrend", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"keltner_multiplier": 1.5, "keltner_period": 20, "stop_atr_mult": 1.5, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 3.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "keltner_multiplier": ParameterSpec(1.0, 3.0, 0.1),
            "keltner_period": ParameterSpec(10, 50, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "supertrend_multiplier": ParameterSpec(1.0, 5.0, 0.1),
            "supertrend_period": ParameterSpec(5, 20, 1),
            "tp_atr_mult": ParameterSpec(2.0, 6.0, 0.1),
            "warmup": ParameterSpec(30, 100, 1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        keltner = indicators["keltner"]
        supertrend = indicators["supertrend"]
        atr = np.nan_to_num(indicators["atr"])
        
        close = np.nan_to_num(df["close"].values)
        keltner_lower = np.nan_to_num(keltner["lower"])
        keltner_upper = np.nan_to_num(keltner["upper"])
        supertrend_line = np.nan_to_num(supertrend["supertrend"])
        supertrend_direction = np.nan_to_num(supertrend["direction"])
        
        # Entry conditions
        entry_condition = (close < keltner_lower) & (supertrend_line < close)
        
        # Exit conditions
        exit_condition = close > keltner_upper
        
        # Initialize signals
        short_signal = np.zeros_like(close, dtype=float)
        short_signal[entry_condition] = -1.0
        
        # Set exit signals
        exit_mask = exit_condition
        short_signal[exit_mask] = 0.0
        
        # Stop loss and take profit conditions
        entry_mask = short_signal == -1.0
        stop_loss = close - (atr * params["stop_atr_mult"])
        take_profit = close - (atr * params["tp_atr_mult"])
        
        # Apply trailing stop logic
        for i in range(1, len(close)):
            if entry_mask[i] and entry_mask[i-1]:
                # If already in position, check for stop loss or take profit
                if close[i] >= stop_loss[i-1]:
                    short_signal[i] = 0.0
                elif close[i] <= take_profit[i-1]:
                    short_signal[i] = 0.0
            elif entry_mask[i]:
                # New entry
                pass
        
        signals = pd.Series(short_signal, index=df.index, dtype=np.float64)
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals