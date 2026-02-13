from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mean_reversion_bollinger_rsi_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(50, 90, 1),
            "rsi_oversold": ParameterSpec(10, 50, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
            "warmup": ParameterSpec(10, 100, 1),
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
        rsi = np.nan_to_num(indicators["rsi"])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_middle = np.nan_to_num(indicators["bollinger"]["middle"])
        bb_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        atr = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)
        open_ = np.nan_to_num(df["open"].values)
        
        # RSI overbought threshold
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_period = params.get("rsi_period", 14)
        
        # Entry conditions
        # Price touches upper band with RSI in overbought zone
        price_touches_upper = close >= bb_upper * 0.99
        rsi_overbought_zone = rsi > rsi_overbought
        # RSI shows exhaustion (weakens or reverses)
        rsi_exhaustion = rsi < rsi_overbought - 5
        # Bullish candle confirmation
        bullish_candle = close > open_
        # RSI trend confirmation
        rsi_shifted = np.roll(rsi, 1)
        rsi_trend_confirmed = rsi_shifted > rsi
        
        entry_condition = (
            price_touches_upper &
            rsi_overbought_zone &
            rsi_exhaustion &
            bullish_candle &
            rsi_trend_confirmed
        )
        
        # Exit when price returns to middle band
        exit_condition = close <= bb_middle
        
        # Create signal array
        entry_signals = np.where(entry_condition, 1.0, 0.0)
        exit_signals = np.where(exit_condition, 0.0, 1.0)
        
        # Combine signals
        signals = pd.Series(entry_signals, index=df.index)
        # Apply exit signals
        signals = signals.where(~exit_condition, 0.0)
        
        # Apply warmup
        signals.iloc[:warmup] = 0.0
        
        return signals