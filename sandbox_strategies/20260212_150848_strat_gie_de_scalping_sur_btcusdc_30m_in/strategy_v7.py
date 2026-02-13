from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="scalping_bollinger_vwap_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "vwap", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"stop_atr_mult": 1.0, "tp_atr_mult": 1.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                value_type=float,
                min_value=0.5,
                max_value=2.0,
                step=0.1
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                value_type=float,
                min_value=1.0,
                max_value=3.0,
                step=0.1
            ),
            "warmup": ParameterSpec(
                name="warmup",
                value_type=int,
                min_value=20,
                max_value=100,
                step=10
            )
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
        vwap = np.nan_to_num(indicators["vwap"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Bollinger bands
        bb_upper = np.nan_to_num(bb["upper"])
        bb_lower = np.nan_to_num(bb["lower"])
        bb_middle = np.nan_to_num(bb["middle"])
        
        # OHLC data
        close = np.nan_to_num(df["close"].values)
        open_ = np.nan_to_num(df["open"].values)
        
        # Parameters
        stop_atr_mult = params.get("stop_atr_mult", 1.0)
        tp_atr_mult = params.get("tp_atr_mult", 1.5)
        warmup = int(params.get("warmup", 50))
        
        # Initialize positions
        position = 0
        entry_price = 0.0
        stop_level = 0.0
        tp_level = 0.0
        
        # Warmup
        signals.iloc[:warmup] = 0.0
        
        # Generate signals
        for i in range(warmup, len(df)):
            if position == 0:
                # Long entry
                long_condition = (close[i] < bb_lower[i]) & (vwap[i] < close[i]) & (close[i] > open_[i])
                if long_condition:
                    position = 1
                    entry_price = close[i]
                    stop_level = entry_price - (stop_atr_mult * atr[i])
                    tp_level = entry_price + (tp_atr_mult * atr[i])
                    signals.iloc[i] = 1.0
                # Short entry
                short_condition = (close[i] > bb_upper[i]) & (vwap[i] > close[i]) & (close[i] < open_[i])
                if short_condition:
                    position = -1
                    entry_price = close[i]
                    stop_level = entry_price + (stop_atr_mult * atr[i])
                    tp_level = entry_price - (tp_atr_mult * atr[i])
                    signals.iloc[i] = -1.0
            else:
                # Exit conditions
                if position > 0:
                    # Long exit
                    if close[i] >= tp_level or close[i] <= stop_level:
                        position = 0
                        signals.iloc[i] = 0.0
                else:
                    # Short exit
                    if close[i] <= tp_level or close[i] >= stop_level:
                        position = 0
                        signals.iloc[i] = 0.0
        
        return signals