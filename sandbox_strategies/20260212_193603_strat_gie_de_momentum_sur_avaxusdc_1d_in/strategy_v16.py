from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="momentum_avaxusdc_1d")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "macd", "atr"]

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
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        rsi = np.nan_to_num(indicators["rsi"])
        macd = indicators["macd"]
        atr = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)
        
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        warmup = int(params.get("warmup", 50))
        
        # Prepare signals
        signals.iloc[:warmup] = 0.0
        
        # Entry conditions
        rsi_long_condition = (rsi > rsi_oversold) & (np.roll(rsi, 1) <= rsi_oversold)
        macd_long_condition = (macd["histogram"] > 0) & (np.roll(macd["histogram"], 1) <= 0)
        long_condition = rsi_long_condition & macd_long_condition
        
        rsi_short_condition = (rsi < rsi_overbought) & (np.roll(rsi, 1) >= rsi_overbought)
        macd_short_condition = (macd["histogram"] < 0) & (np.roll(macd["histogram"], 1) >= 0)
        short_condition = rsi_short_condition & macd_short_condition
        
        # Exit conditions
        exit_long_condition = (rsi > rsi_overbought) | (rsi < rsi_oversold) | ((rsi > 50) & (close < np.roll(close, 1))) | ((rsi < 50) & (close > np.roll(close, 1)))
        exit_short_condition = (rsi > rsi_overbought) | (rsi < rsi_oversold) | ((rsi > 50) & (close < np.roll(close, 1))) | ((rsi < 50) & (close > np.roll(close, 1)))
        
        # Generate signals
        long_entries = np.where(long_condition, 1.0, 0.0)
        short_entries = np.where(short_condition, -1.0, 0.0)
        
        # Apply exits
        exit_long = np.where(exit_long_condition, 0.0, 1.0)
        exit_short = np.where(exit_short_condition, 0.0, 1.0)
        
        # Combine entries and exits
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        signals.iloc[:warmup] = 0.0
        
        # Long signals
        long_signal = np.zeros_like(close)
        long_signal[long_condition] = 1.0
        signals.iloc[long_condition] = 1.0
        
        # Short signals
        short_signal = np.zeros_like(close)
        short_signal[short_condition] = -1.0
        signals.iloc[short_condition] = -1.0
        
        return signals