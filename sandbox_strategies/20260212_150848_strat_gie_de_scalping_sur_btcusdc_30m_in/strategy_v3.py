from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btc_vwap_bollinger_atr_scalper")

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
                param_type="float",
                min_value=0.5,
                max_value=2.0,
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                param_type="float",
                min_value=1.0,
                max_value=3.0,
                step=0.1,
            ),
            "warmup": ParameterSpec(
                name="warmup",
                param_type="int",
                min_value=20,
                max_value=100,
                step=10,
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
        
        # Bollinger Bands
        bb_upper = np.nan_to_num(bb["upper"])
        bb_lower = np.nan_to_num(bb["lower"])
        
        # Close prices
        close = np.nan_to_num(df["close"].values)
        
        # Entry conditions
        long_condition = (close <= bb_lower) & (close >= vwap)
        short_condition = (close >= bb_upper) & (close <= vwap)
        
        # Generate signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0
        
        return signals