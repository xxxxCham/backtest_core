from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_breakout_atr_trail_v2")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "leverage": 1,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.5,
            "warmup": 50
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "leverage": ParameterSpec(type_=(int, float), default=1, min_=1, max_=10, desc="Leverage"),
            "stop_atr_mult": ParameterSpec(type_=float, default=1.5, min_=0.1, max_=5.0, desc="ATR stop multiplier"),
            "tp_atr_mult": ParameterSpec(type_=float, default=3.5, min_=0.1, max_=10.0, desc="ATR take-profit multiplier"),
            "warmup": ParameterSpec(type_=int, default=50, min_=10, max_=200, desc="Warmup period")
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        atr = np.nan_to_num(indicators["atr"])
        bb = indicators["bollinger"]
        volume = df["volume"].values
        median_volume = pd.Series(volume).rolling(window=20).mean().values
        median_volume = np.nan_to_num(median_volume)
        
        warmup = int(params["warmup"])
        signals.iloc[:warmup] = 0.0
        
        upper = np.nan_to_num(bb["upper"])
        lower = np.nan_to_num(bb["lower"])
        close = df["close"].values
        
        entry_long_mask = (close > upper) & (volume > median_volume)
        entry_short_mask = (close < lower) & (volume > median_volume)
        signals[entry_long_mask] = 1.0
        signals[entry_short_mask] = -1.0
        
        exit_mask = (close < lower) | (close > upper)
        signals[exit_mask] = 0.0
        
        sl_long = close - params["stop_atr_mult"] * atr
        tp_long = close + params["tp_atr_mult"] * atr
        sl_short = close + params["stop_atr_mult"] * atr
        tp_short = close - params["tp_atr_mult"] * atr
        
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        
        long_entries = signals == 1.0
        short_entries = signals == -1.0
        
        df.loc[long_entries, "bb_stop_long"] = sl_long[long_entries]
        df.loc[long_entries, "bb_tp_long"] = tp_long[long_entries]
        df.loc[short_entries, "bb_stop_short"] = sl_short[short_entries]
        df.loc[short_entries, "bb_tp_short"] = tp_short[short_entries]
        
        return signals