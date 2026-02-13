from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="donchian_williams_mean_reversion")

    @property
    def required_indicators(self) -> List[str]:
        return ["donchian", "williams_r", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"donchian_period": 20, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50, "williams_r_overbought": -20, "williams_r_period": 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "donchian_period": ParameterSpec(param_type="int", min_value=5, max_value=50, step=5),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=3.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=5.0, step=0.5),
            "warmup": ParameterSpec(param_type="int", min_value=30, max_value=100, step=10),
            "williams_r_overbought": ParameterSpec(param_type="int", min_value=-50, max_value=-5, step=5),
            "williams_r_period": ParameterSpec(param_type="int", min_value=5, max_value=30, step=5),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        donchian_period = int(params.get("donchian_period", 20))
        stop_atr_mult = float(params.get("stop_atr_mult", 1.5))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.0))
        williams_r_overbought = float(params.get("williams_r_overbought", -20))
        
        donchian = indicators["donchian"]
        upper_band = np.nan_to_num(donchian["upper"])
        middle_band = np.nan_to_num(donchian["middle"])
        lower_band = np.nan_to_num(donchian["lower"])
        
        williams_r = np.nan_to_num(indicators["williams_r"])
        atr = np.nan_to_num(indicators["atr"])
        
        price = np.nan_to_num(df["close"].values)
        
        # Entry condition: price touches upper band and Williams %R is overbought
        entry_condition = (np.abs(price - upper_band) < 1e-8) & (williams_r < williams_r_overbought)
        
        # Exit condition: price crosses back to middle band or continues in overbought territory
        exit_condition = (np.abs(price - middle_band) < 1e-8) | (williams_r < williams_r_overbought)
        
        # Generate signals
        entry_signals = np.where(entry_condition, 1.0, 0.0)
        exit_signals = np.where(exit_condition, -1.0, 0.0)
        
        # Combine signals
        signals = pd.Series(entry_signals, index=df.index, dtype=np.float64)
        signals = signals.where(~(exit_signals == -1.0), -1.0)
        
        return signals