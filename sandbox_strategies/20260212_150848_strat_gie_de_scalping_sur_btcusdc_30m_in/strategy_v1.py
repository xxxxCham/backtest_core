from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btc_vwap_bollinger_scalper")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "vwap", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"atr_period": 14, "bollinger_period": 20, "bollinger_std_dev": 2, "stop_atr_mult": 1.0, "tp_atr_mult": 1.5, "vwap_period": 14, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "atr_period": ParameterSpec("atr_period", 5, 30, 1),
            "bollinger_period": ParameterSpec("bollinger_period", 10, 50, 1),
            "bollinger_std_dev": ParameterSpec("bollinger_std_dev", 1.0, 3.0, 0.5),
            "stop_atr_mult": ParameterSpec("stop_atr_mult", 0.5, 2.0, 0.5),
            "tp_atr_mult": ParameterSpec("tp_atr_mult", 1.0, 3.0, 0.5),
            "vwap_period": ParameterSpec("vwap_period", 5, 30, 1),
            "warmup": ParameterSpec("warmup", 20, 100, 10),
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
        
        # Get Bollinger bands
        lower_bb = np.nan_to_num(bb["lower"])
        upper_bb = np.nan_to_num(bb["upper"])
        middle_bb = np.nan_to_num(bb["middle"])
        
        # Get price
        close = np.nan_to_num(df["close"].values)
        
        # VWAP series for previous value comparison
        vwap_shifted = np.roll(vwap, 1)
        vwap_shifted[0] = vwap[0]
        
        # Price crossing conditions
        # Long entry: price crosses above lower Bollinger Band
        # AND VWAP is above its previous value
        # AND price is above VWAP
        long_condition = (
            (close > lower_bb) & 
            (close < np.roll(close, 1)) &
            (vwap > vwap_shifted) &
            (close > vwap)
        )
        
        # Short entry: price crosses below upper Bollinger Band
        # AND VWAP is below its previous value
        # AND price is below VWAP
        short_condition = (
            (close < upper_bb) & 
            (close > np.roll(close, 1)) &
            (vwap < vwap_shifted) &
            (close < vwap)
        )
        
        # Generate signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals