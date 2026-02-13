from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="momentum_btcusdc_30m_short_revised")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "stop_atr_mult": ParameterSpec(1.0, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 4.0, 0.1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        rsi_overbought = params["rsi_overbought"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        bb_upper = np.nan_to_num(bb["upper"])
        bb_middle = np.nan_to_num(bb["middle"])
        bb_lower = np.nan_to_num(bb["lower"])
        atr = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)
        high = np.nan_to_num(df["high"].values)
        low = np.nan_to_num(df["low"].values)

        # Entry conditions for short
        rsi_crossed_overbought = (rsi[1:] > rsi_overbought) & (rsi[:-1] <= rsi_overbought)
        price_above_bb_upper = close[1:] > bb_upper[1:]
        mfi_confirm_short = np.nan_to_num(indicators["mfi"])[1:] < 50

        short_entry = rsi_crossed_overbought & price_above_bb_upper & mfi_confirm_short
        entry_indices = np.where(short_entry)[0] + 1  # +1 because we compare with previous bar

        # Assign signals
        for idx in entry_indices:
            signals.iloc[idx] = -1.0  # Short signal

        return signals