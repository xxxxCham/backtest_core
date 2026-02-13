from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="supertrend_adx_atr_trend_following")

    @property
    def required_indicators(self) -> List[str]:
        return ["supertrend", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"adx_period": 14, "adx_threshold": 25, "stop_atr_mult": 1.0, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec("adx_period", 5, 30, 1),
            "adx_threshold": ParameterSpec("adx_threshold", 10, 50, 1),
            "stop_atr_mult": ParameterSpec("stop_atr_mult", 0.5, 3.0, 0.1),
            "supertrend_multiplier": ParameterSpec("supertrend_multiplier", 1.0, 5.0, 0.1),
            "supertrend_period": ParameterSpec("supertrend_period", 5, 20, 1),
            "tp_atr_mult": ParameterSpec("tp_atr_mult", 1.0, 5.0, 0.1),
            "warmup": ParameterSpec("warmup", 20, 100, 5),
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
        supertrend = np.nan_to_num(indicators["supertrend"]["supertrend"])
        direction = np.nan_to_num(indicators["supertrend"]["direction"])
        adx = np.nan_to_num(indicators["adx"]["adx"])
        atr = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)
        
        # Entry conditions
        # Long entry: supertrend above price AND adx above threshold
        entry_long = (direction > 0) & (supertrend < close) & (adx > params["adx_threshold"])
        
        # Exit conditions
        # Exit: supertrend below price OR adx below threshold
        exit_long = (direction < 0) | (adx < 20)
        
        # Generate signals
        in_position = False
        position_entry_price = 0.0
        
        for i in range(len(signals)):
            if entry_long[i] and not in_position:
                signals[i] = 1.0
                in_position = True
                position_entry_price = close[i]
            elif exit_long[i] and in_position:
                signals[i] = 0.0
                in_position = False
            elif in_position:
                # Check for take profit or stop loss
                tp_level = position_entry_price + (params["tp_atr_mult"] * atr[i])
                sl_level = position_entry_price - (params["stop_atr_mult"] * atr[i])
                if close[i] >= tp_level or close[i] <= sl_level:
                    signals[i] = 0.0
                    in_position = False
                else:
                    signals[i] = 1.0
            else:
                signals[i] = 0.0
                
        return signals