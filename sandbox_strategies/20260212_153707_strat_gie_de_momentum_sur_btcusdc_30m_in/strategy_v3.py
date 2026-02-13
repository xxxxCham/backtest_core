from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btc_rsi_macd_atr_momentum")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "macd", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 90, 1),
            "rsi_oversold": ParameterSpec(10, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        rsi = np.nan_to_num(indicators["rsi"])
        macd = indicators["macd"]
        macd_hist = np.nan_to_num(macd["histogram"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Short entry conditions
        rsi_crossed_above = (np.roll(rsi, 1) < rsi_overbought) & (rsi >= rsi_overbought)
        macd_hist_negative = macd_hist < 0
        macd_hist_falling = macd_hist < np.roll(macd_hist, 1)
        short_entry = rsi_crossed_above & macd_hist_negative & macd_hist_falling
        
        # Exit conditions
        rsi_crossed_below = (np.roll(rsi, 1) > rsi_overbought) & (rsi <= rsi_overbought)
        macd_hist_crossed_zero = (np.roll(macd_hist, 1) < 0) & (macd_hist > 0)
        exit_condition = rsi_crossed_below | macd_hist_crossed_zero
        
        # Set signals
        short_positions = np.zeros_like(rsi, dtype=bool)
        for i in range(1, len(rsi)):
            if short_entry[i]:
                short_positions[i] = True
            elif short_positions[i-1] and exit_condition[i]:
                short_positions[i] = False
            else:
                short_positions[i] = short_positions[i-1]
        
        signals.loc[short_positions] = -1.0
        
        return signals