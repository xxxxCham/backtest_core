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
        return ["macd", "roc", "atr", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"macd_fast": 12, "macd_signal": 9, "macd_slow": 26, "roc_period": 10, "rsi_period": 14, "stop_atr_mult": 2.0, "tp_atr_mult": 4.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "macd_fast": ParameterSpec(5, 20, 1),
            "macd_signal": ParameterSpec(5, 15, 1),
            "macd_slow": ParameterSpec(20, 40, 1),
            "roc_period": ParameterSpec(5, 20, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.5),
            "tp_atr_mult": ParameterSpec(2.0, 10.0, 0.5),
            "warmup": ParameterSpec(20, 100, 10)
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
        roc = np.nan_to_num(indicators["roc"])
        atr = np.nan_to_num(indicators["atr"])
        rsi = np.nan_to_num(indicators["rsi"])
        
        # Get histogram values
        macd_histogram = np.nan_to_num(macd["histogram"])
        
        # Entry conditions
        # Long entry: MACD histogram positive and accelerating, ROC positive, RSI not overbought
        long_condition = (
            (macd_histogram > 0) &
            (macd_histogram > np.roll(macd_histogram, 1)) &
            (roc > 0) &
            (rsi < 70)
        )
        
        # Short entry: MACD histogram negative and accelerating, ROC negative, RSI not oversold
        short_condition = (
            (macd_histogram < 0) &
            (macd_histogram < np.roll(macd_histogram, 1)) &
            (roc < 0) &
            (rsi > 30)
        )
        
        # Exit conditions: momentum lost (histogram changes sign or ROC changes sign)
        exit_long = (macd_histogram < 0) & (roc < 0)
        exit_short = (macd_histogram > 0) & (roc > 0)
        
        # Set signals
        long_entries = long_condition
        short_entries = short_condition
        long_exits = exit_long
        short_exits = exit_short
        
        # Apply signals
        signals[long_entries] = 1.0
        signals[short_entries] = -1.0
        signals[long_exits] = 0.0
        signals[short_exits] = 0.0
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals