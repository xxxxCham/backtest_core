from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="ema_rsi_bollinger_sc_continuation")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "bollinger_period": 20,
            "bollinger_std_dev": 2,
            "ema_fast": 9,
            "ema_mid": 21,
            "ema_slow": 50,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "bollinger_period": ParameterSpec(min=5, max=100, default=20, description="Bollinger period"),
            "bollinger_std_dev": ParameterSpec(min=0.5, max=5.0, default=2.0, description="Bollinger standard deviation"),
            "ema_fast": ParameterSpec(min=5, max=30, default=9, description="Fast EMA period"),
            "ema_mid": ParameterSpec(min=10, max=50, default=21, description="Mid EMA period"),
            "ema_slow": ParameterSpec(min=30, max=200, default=50, description="Slow EMA period"),
            "rsi_overbought": ParameterSpec(min=60, max=90, default=70, description="RSI overbought level"),
            "rsi_oversold": ParameterSpec(min=10, max=40, default=30, description="RSI oversold level"),
            "rsi_period": ParameterSpec(min=5, max=30, default=14, description="RSI period"),
            "stop_atr_mult": ParameterSpec(min=0.5, max=5.0, default=1.5, description="Stop loss ATR multiplier"),
            "tp_atr_mult": ParameterSpec(min=1.0, max=10.0, default=3.0, description="Take profit ATR multiplier"),
            "warmup": ParameterSpec(min=0, max=200, default=50, description="Warmup bars to ignore"),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        # Prepare output series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Extract price series
        close = df["close"].values.astype(np.float64)

        # Extract indicators and sanitize
        ema = np.nan_to_num(indicators["ema"]).astype(np.float64)
        rsi = np.nan_to_num(indicators["rsi"]).astype(np.float64)

        bb = indicators["bollinger"]
        lower = np.nan_to_num(bb["lower"]).astype(np.float64)
        upper = np.nan_to_num(bb["upper"]).astype(np.float64)

        atr = np.nan_to_num(indicators["atr"]).astype(np.float64)

        # Parameters
        overbought = float(params["rsi_overbought"])
        oversold = float(params["rsi_oversold"])

        # Shifted RSI for previous bar (first value becomes NaN)
        rsi_prev = np.concatenate([[np.nan], rsi[:-1]])

        # Entry conditions
        entry_long = (close < ema) & (close <= lower) & (rsi_prev <= oversold) & (rsi > oversold)
        entry_short = (close > ema) & (close >= upper) & (rsi_prev >= overbought) & (rsi < overbought)

        # Exit conditions
        exit_long = (close >= upper) | (rsi >= overbought) | (close < ema)
        exit_short = (close <= lower) | (rsi <= oversold) | (close > ema)

        # Position tracking
        position = 0.0
        for i in range(len(df)):
            if i < warmup:
                continue

            if position == 0.0:
                if entry_long[i]:
                    position = 1.0
                elif entry_short[i]:
                    position = -1.0
            elif position == 1.0:
                if exit_long[i]:
                    position = 0.0
            elif position == -1.0:
                if exit_short[i]:
                    position = 0.0

            signals.iloc[i] = position

        return signals