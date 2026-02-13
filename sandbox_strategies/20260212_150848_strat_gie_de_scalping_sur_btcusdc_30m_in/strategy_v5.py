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
                value_range=(0.5, 2.0),
                step=0.1,
                default=1.0,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                value_range=(1.0, 3.0),
                step=0.1,
                default=1.5,
            ),
            "warmup": ParameterSpec(
                name="warmup",
                value_range=(20, 100),
                step=10,
                default=50,
            ),
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
        bb = indicators["bollinger"]
        vwap = np.nan_to_num(indicators["vwap"])
        atr = np.nan_to_num(indicators["atr"])
        
        close = np.nan_to_num(df["close"].values)
        open_ = np.nan_to_num(df["open"].values)
        
        bb_upper = np.nan_to_num(bb["upper"])
        bb_lower = np.nan_to_num(bb["lower"])
        
        # Entry conditions
        # Long entry: close crosses below bb_lower and vwap is below close
        close_below_lower = close < bb_lower
        vwap_below_close = vwap < close
        
        # Short entry: close crosses above bb_upper and vwap is above close
        close_above_upper = close > bb_upper
        vwap_above_close = vwap > close
        
        # Detect crossovers
        prev_close = np.roll(close, 1)
        prev_bb_lower = np.roll(bb_lower, 1)
        prev_bb_upper = np.roll(bb_upper, 1)
        
        cross_below = (prev_close >= prev_bb_lower) & (close_below_lower)
        cross_above = (prev_close <= prev_bb_upper) & (close_above_upper)
        
        # Combine conditions
        long_condition = cross_below & vwap_below_close
        short_condition = cross_above & vwap_above_close
        
        # Generate signals
        signals.loc[long_condition] = 1.0
        signals.loc[short_condition] = -1.0
        
        return signals