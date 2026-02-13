from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="keltner_supertrend_atr_breakout_enhanced")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "supertrend", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"keltner_multiplier": 1.5, "keltner_period": 20, "stop_atr_mult": 2.0, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "keltner_multiplier": ParameterSpec(1.0, 3.0, 0.1),
            "keltner_period": ParameterSpec(10, 50, 1),
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
        price = np.nan_to_num(df["close"].values)
        keltner = indicators["keltner"]
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_lower = np.nan_to_num(keltner["lower"])
        keltner_middle = np.nan_to_num(keltner["middle"])
        supertrend = indicators["supertrend"]
        supertrend_value = np.nan_to_num(supertrend["supertrend"])
        supertrend_direction = np.nan_to_num(supertrend["direction"])
        atr = np.nan_to_num(indicators["atr"])
        keltner_multiplier = params.get("keltner_multiplier", 1.5)
        keltner_period = params.get("keltner_period", 20)
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        supertrend_multiplier = params.get("supertrend_multiplier", 3.0)
        supertrend_period = params.get("supertrend_period", 10)
        tp_atr_mult = params.get("tp_atr_mult", 5.0)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        long_entry = (price > keltner_upper) & (supertrend_direction > 0) & (price > np.roll(price, 1))
        short_entry = (price < keltner_lower) & (supertrend_direction < 0) & (price < np.roll(price, 1))
        long_signal = np.zeros_like(price, dtype=bool)
        short_signal = np.zeros_like(price, dtype=bool)
        long_signal[long_entry] = True
        short_signal[short_entry] = True
        long_positions = np.zeros_like(price, dtype=bool)
        short_positions = np.zeros_like(price, dtype=bool)
        entry_price_long = np.full_like(price, np.nan)
        entry_price_short = np.full_like(price, np.nan)
        highest_high = np.full_like(price, np.nan)
        lowest_low = np.full_like(price, np.nan)
        for i in range(1, len(price)):
            if long_signal[i]:
                entry_price_long[i] = price[i]
                highest_high[i] = price[i]
            elif short_signal[i]:
                entry_price_short[i] = price[i]
                lowest_low[i] = price[i]
            if not np.isnan(entry_price_long[i-1]):
                highest_high[i] = max(highest_high[i-1], price[i])
                if price[i] <= (entry_price_long[i-1] - stop_atr_mult * atr[i-1]) or price[i] >= (entry_price_long[i-1] + tp_atr_mult * atr[i-1]):
                    long_positions[i] = True
                elif price[i] < highest_high[i-1] - atr[i-1]:
                    long_positions[i] = True
            if not np.isnan(entry_price_short[i-1]):
                lowest_low[i] = min(lowest_low[i-1], price[i])
                if price[i] >= (entry_price_short[i-1] + stop_atr_mult * atr[i-1]) or price[i] <= (entry_price_short[i-1] - tp_atr_mult * atr[i-1]):
                    short_positions[i] = True
                elif price[i] > lowest_low[i-1] + atr[i-1]:
                    short_positions[i] = True
        signals[long_positions] = 1.0
        signals[short_positions] = -1.0
        return signals