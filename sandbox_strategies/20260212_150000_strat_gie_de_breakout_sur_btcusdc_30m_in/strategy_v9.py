from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="keltner_supertrend_breakout")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "supertrend", "atr", "ema"]

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
        
        # Extract indicators
        keltner = indicators["keltner"]
        supertrend = indicators["supertrend"]
        atr = np.nan_to_num(indicators["atr"])
        ema = np.nan_to_num(indicators["ema"])
        
        # Keltner bands
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_lower = np.nan_to_num(keltner["lower"])
        keltner_middle = np.nan_to_num(keltner["middle"])
        
        # Supertrend
        st_line = np.nan_to_num(supertrend["supertrend"])
        st_direction = np.nan_to_num(supertrend["direction"])
        
        # EMA 20
        ema_20 = np.nan_to_num(ema)
        
        # Parameters
        keltner_mult = params.get("keltner_multiplier", 1.5)
        keltner_period = params.get("keltner_period", 20)
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        supertrend_mult = params.get("supertrend_multiplier", 3.0)
        supertrend_period = params.get("supertrend_period", 10)
        tp_atr_mult = params.get("tp_atr_mult", 5.0)
        warmup = int(params.get("warmup", 50))
        
        # Set warmup period
        signals.iloc[:warmup] = 0.0
        
        # Price
        close = np.nan_to_num(df["close"].values)
        
        # Entry conditions
        # Long entry: price breaks above Keltner upper band AND Supertrend line is below price AND price is above 20-period EMA
        long_condition = (
            (close > keltner_upper) &
            (st_line < close) &
            (close > ema_20)
        )
        
        # Short entry: price breaks below Keltner lower band AND Supertrend line is above price AND price is below 20-period EMA
        short_condition = (
            (close < keltner_lower) &
            (st_line > close) &
            (close < ema_20)
        )
        
        # Generate signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0
        
        return signals