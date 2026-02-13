from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_mean_reversion_bollinger_rsi_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(50, 90, 1),
            "rsi_oversold": ParameterSpec(10, 50, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.5),
            "tp_atr_mult": ParameterSpec(2.0, 10.0, 0.5),
            "warmup": ParameterSpec(10, 100, 10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        rsi_overbought = params.get("rsi_overbought", 70)
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 5.0)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])
        bb = indicators["bollinger"]
        lower = np.nan_to_num(bb["lower"])
        middle = np.nan_to_num(bb["middle"])
        close = np.nan_to_num(df["close"].values)
        price_touches_lower = np.abs(close - lower) < (0.001 * close)
        rsi_overbought_condition = rsi > rsi_overbought
        entry_condition = price_touches_lower & rsi_overbought_condition
        exit_condition = close > middle
        entry_indices = np.where(entry_condition)[0]
        exit_indices = np.where(exit_condition)[0]
        for i in entry_indices:
            if i < len(signals) - 1:
                signals.iloc[i] = -1.0
                stop_loss = close[i] + stop_atr_mult * atr[i]
                take_profit = close[i] - tp_atr_mult * atr[i]
                for j in range(i + 1, len(signals)):
                    if close[j] >= stop_loss or close[j] <= take_profit:
                        signals.iloc[j] = 0.0
                        break
                    if close[j] > middle:
                        signals.iloc[j] = 0.0
                        break
        return signals