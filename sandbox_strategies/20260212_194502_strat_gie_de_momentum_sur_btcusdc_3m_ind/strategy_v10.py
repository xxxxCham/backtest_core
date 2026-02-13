from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_momentum_three_indicator")

    @property
    def required_indicators(self) -> List[str]:
        return ["macd", "roc", "atr", "rsi", "bollinger"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"bb_period": 20, "bb_std_dev": 2, "macd_fast": 12, "macd_signal": 9, "macd_slow": 26, "roc_period": 10, "rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 2.0, "tp_atr_mult": 4.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "bb_period": ParameterSpec(10, 50, 20),
            "bb_std_dev": ParameterSpec(1.0, 3.0, 2.0),
            "macd_fast": ParameterSpec(5, 20, 12),
            "macd_signal": ParameterSpec(3, 15, 9),
            "macd_slow": ParameterSpec(15, 50, 26),
            "roc_period": ParameterSpec(5, 30, 10),
            "rsi_overbought": ParameterSpec(60, 90, 70),
            "rsi_oversold": ParameterSpec(10, 40, 30),
            "rsi_period": ParameterSpec(5, 30, 14),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 2.0),
            "tp_atr_mult": ParameterSpec(2.0, 10.0, 4.5),
            "warmup": ParameterSpec(10, 100, 50),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        macd = indicators["macd"]
        roc = indicators["roc"]
        atr = indicators["atr"]
        rsi = indicators["rsi"]
        bb = indicators["bollinger"]
        
        # Extract arrays from dicts
        macd_hist = np.nan_to_num(macd["histogram"])
        roc_values = np.nan_to_num(roc)
        rsi_values = np.nan_to_num(rsi)
        bb_upper = np.nan_to_num(bb["upper"])
        bb_lower = np.nan_to_num(bb["lower"])
        close = np.nan_to_num(df["close"].values)
        
        # Compute momentum acceleration
        macd_hist_diff = np.diff(macd_hist, prepend=np.nan)
        
        # Entry conditions
        long_condition = (
            (macd_hist > 0) &
            (macd_hist_diff > 0) &
            (roc_values > 0) &
            (rsi_values < params["rsi_overbought"]) &
            (close > bb_upper)
        )
        
        short_condition = (
            (macd_hist < 0) &
            (macd_hist_diff < 0) &
            (roc_values < 0) &
            (rsi_values > params["rsi_oversold"]) &
            (close < bb_lower)
        )
        
        # Exit conditions
        exit_long_condition = (macd_hist < 0) & (roc_values < 0)
        exit_short_condition = (macd_hist > 0) & (roc_values > 0)
        
        # Generate signals
        long_signals = np.where(long_condition, 1.0, 0.0)
        short_signals = np.where(short_condition, -1.0, 0.0)
        exit_signals = np.where(exit_long_condition | exit_short_condition, 0.0, 0.0)
        
        # Combine signals
        signals = pd.Series(long_signals - short_signals, index=df.index, dtype=np.float64)
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals