from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bolinger_supertrend_atr_rsi_breakout")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "supertrend", "atr", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 90, 1),
            "rsi_oversold": ParameterSpec(10, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 6.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1),
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
        st = indicators["supertrend"]
        atr = np.nan_to_num(indicators["atr"])
        rsi = np.nan_to_num(indicators["rsi"])
        
        # Extract BB components
        upper_bb = np.nan_to_num(bb["upper"])
        lower_bb = np.nan_to_num(bb["lower"])
        middle_bb = np.nan_to_num(bb["middle"])
        
        # Extract Supertrend components
        st_direction = np.nan_to_num(st["direction"])
        st_value = np.nan_to_num(st["supertrend"])
        
        # Volume
        volume = np.nan_to_num(df["volume"])
        avg_volume = np.nan_to_num(pd.Series(volume).rolling(20).mean().values)
        
        # Entry conditions
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        
        # Long entry
        long_condition = (
            (df["close"].values > upper_bb) &
            (st_direction > 0) &
            (rsi < rsi_overbought) &
            (volume > avg_volume * 1.5)
        )
        
        # Short entry
        short_condition = (
            (df["close"].values < lower_bb) &
            (st_direction < 0) &
            (rsi > rsi_oversold) &
            (volume > avg_volume * 1.5)
        )
        
        # Generate signals
        signals.loc[long_condition] = 1.0
        signals.loc[short_condition] = -1.0
        
        return signals