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
        return {"atr_period": 14, "keltner_multiplier": 1.5, "keltner_period": 20, "stop_atr_mult": 2.0, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "atr_period": ParameterSpec(10, 30, 1),
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
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Extract indicators
        keltner = indicators["keltner"]
        upper_keltner = np.nan_to_num(keltner["upper"])
        lower_keltner = np.nan_to_num(keltner["lower"])
        supertrend = indicators["supertrend"]
        supertrend_line = np.nan_to_num(supertrend["supertrend"])
        supertrend_direction = np.nan_to_num(supertrend["direction"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Price array
        close = np.nan_to_num(df["close"].values)
        
        # Entry conditions
        price_above_upper = close > upper_keltner
        price_below_lower = close < lower_keltner
        trend_bullish = supertrend_direction > 0
        trend_bearish = supertrend_direction < 0
        
        # Long entry: price crosses above upper Keltner AND supertrend is bullish
        long_entry = price_above_upper & trend_bullish
        
        # Short entry: price crosses below lower Keltner AND supertrend is bearish
        short_entry = price_below_lower & trend_bearish
        
        # Signal generation
        long_signal = np.zeros_like(close, dtype=bool)
        short_signal = np.zeros_like(close, dtype=bool)
        
        # Initialize last entry flags
        last_long_entry = False
        last_short_entry = False
        
        for i in range(1, len(close)):
            # Long signal
            if long_entry[i] and not last_long_entry:
                long_signal[i] = True
                last_long_entry = True
                last_short_entry = False
            # Short signal
            elif short_entry[i] and not last_short_entry:
                short_signal[i] = True
                last_short_entry = True
                last_long_entry = False
        
        # Convert to signals
        signals[long_signal] = 1.0
        signals[short_signal] = -1.0
        
        return signals