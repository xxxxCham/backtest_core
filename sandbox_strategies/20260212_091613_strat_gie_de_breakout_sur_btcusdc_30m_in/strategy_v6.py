from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bolinger_supertrend_atr_rsi_breakout")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "supertrend", "atr", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(50, 100, 1, "Overbought RSI level"),
            "rsi_oversold": ParameterSpec(0, 50, 1, "Oversold RSI level"),
            "rsi_period": ParameterSpec(5, 30, 1, "RSI period"),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1, "Stop loss multiplier for ATR"),
            "tp_atr_mult": ParameterSpec(2.0, 6.0, 0.1, "Take profit multiplier for ATR"),
            "warmup": ParameterSpec(20, 100, 1, "Warmup period for signals"),
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
        upper_band = np.nan_to_num(bb["upper"])
        lower_band = np.nan_to_num(bb["lower"])
        close = np.nan_to_num(df["close"].values)
        volume = np.nan_to_num(df["volume"].values)
        avg_volume = np.nan_to_num(pd.Series(volume).rolling(20).mean().values)
        supertrend_direction = np.nan_to_num(indicators["supertrend"]["direction"])
        atr = np.nan_to_num(indicators["atr"])
        rsi = np.nan_to_num(indicators["rsi"])
        
        # Entry conditions
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        
        # Long entry: close breaks upper band, supertrend up, RSI not overbought, volume confirmed
        long_condition = (
            (close > upper_band) &
            (supertrend_direction > 0) &
            (rsi < rsi_overbought) &
            (volume > avg_volume * 1.5)
        )
        
        # Short entry: close breaks lower band, supertrend down, RSI not oversold, volume confirmed
        short_condition = (
            (close < lower_band) &
            (supertrend_direction < 0) &
            (rsi > rsi_oversold) &
            (volume > avg_volume * 1.5)
        )
        
        # Set signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals