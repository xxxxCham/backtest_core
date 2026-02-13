from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="keltner_supertrend_atr_breakout")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "supertrend", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"keltner_atr_mult": 1.5, "keltner_period": 20, "stop_atr_mult": 2.0, "supertrend_mult": 3.0, "supertrend_period": 10, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "keltner_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "keltner_period": ParameterSpec(10, 50, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.1),
            "supertrend_mult": ParameterSpec(1.0, 5.0, 0.1),
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
        supertrend = np.nan_to_num(indicators["supertrend"]["supertrend"])
        close = np.nan_to_num(df["close"].values)
        atr = np.nan_to_num(indicators["atr"])
        
        # Parameters
        keltner_atr_mult = params.get("keltner_atr_mult", 1.5)
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 5.0)
        warmup = int(params.get("warmup", 50))
        
        # Initialize positions
        position = 0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        
        # Warmup period
        signals.iloc[:warmup] = 0.0
        
        # Generate signals
        for i in range(warmup, len(df)):
            if position == 0:
                # Check for long entry
                if (close[i] > keltner_upper[i]) and (supertrend[i] < close[i]):
                    signals.iloc[i] = 1.0
                    position = 1
                    entry_price = close[i]
                    stop_loss = entry_price - stop_atr_mult * atr[i]
                    take_profit = entry_price + tp_atr_mult * atr[i]
                # Check for short entry
                elif (close[i] < keltner_lower[i]) and (supertrend[i] > close[i]):
                    signals.iloc[i] = -1.0
                    position = -1
                    entry_price = close[i]
                    stop_loss = entry_price + stop_atr_mult * atr[i]
                    take_profit = entry_price - tp_atr_mult * atr[i]
            else:
                # Check for exit conditions
                if position == 1:
                    # Exit long if price hits Keltner upper or stop loss or take profit
                    if (close[i] >= keltner_upper[i]) or (close[i] <= stop_loss) or (close[i] >= take_profit):
                        signals.iloc[i] = 0.0
                        position = 0
                    else:
                        signals.iloc[i] = 1.0
                elif position == -1:
                    # Exit short if price hits Keltner lower or stop loss or take profit
                    if (close[i] <= keltner_lower[i]) or (close[i] >= stop_loss) or (close[i] <= take_profit):
                        signals.iloc[i] = 0.0
                        position = 0
                    else:
                        signals.iloc[i] = -1.0
        
        return signals