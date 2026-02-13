from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bolinger_supertrend_atr_breakout")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "supertrend", "atr", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(50, 100, 1),
            "rsi_oversold": ParameterSpec(0, 50, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 6.0, 0.1),
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
        bb = indicators["bollinger"]
        upper_bb = np.nan_to_num(bb["upper"])
        lower_bb = np.nan_to_num(bb["lower"])
        middle_bb = np.nan_to_num(bb["middle"])
        
        st = indicators["supertrend"]
        supertrend_line = np.nan_to_num(st["supertrend"])
        supertrend_direction = np.nan_to_num(st["direction"])
        
        atr = np.nan_to_num(indicators["atr"])
        rsi = np.nan_to_num(indicators["rsi"])
        price = np.nan_to_num(df["close"].values)
        volume = np.nan_to_num(df["volume"].values)
        
        # Compute average volume over 10 periods
        avg_volume = np.convolve(volume, np.ones(10)/10, mode='valid')
        avg_volume = np.concatenate([np.full(9, np.nan), avg_volume])
        
        # Entry conditions
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        
        # Long entry: price crosses above upper BOLLINGER band, SUPERTREND direction up, RSI below overbought, volume > 1.5x avg
        long_condition = (
            (price > upper_bb) &
            (supertrend_direction > 0) &
            (rsi < rsi_overbought) &
            (volume > 1.5 * avg_volume)
        )
        
        # Short entry: price crosses below lower BOLLINGER band, SUPERTREND direction down, RSI above oversold, volume > 1.5x avg
        short_condition = (
            (price < lower_bb) &
            (supertrend_direction < 0) &
            (rsi > rsi_oversold) &
            (volume > 1.5 * avg_volume)
        )
        
        # Generate signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals