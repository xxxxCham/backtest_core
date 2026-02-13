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
        return {"donchian_period": 20, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50, "williams_r_overbought": -20, "williams_r_oversold": -80}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "donchian_period": ParameterSpec(param_type="int", min_value=5, max_value=50, step=5),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=3.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=5.0, step=0.5),
            "williams_r_overbought": ParameterSpec(param_type="int", min_value=-50, max_value=-10, step=10),
            "williams_r_oversold": ParameterSpec(param_type="int", min_value=-90, max_value=-70, step=10),
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
        williams_r_oversold = float(params.get("williams_r_oversold", -80))
        
        upper_band = np.nan_to_num(indicators["donchian"]["upper"])
        middle_band = np.nan_to_num(indicators["donchian"]["middle"])
        lower_band = np.nan_to_num(indicators["donchian"]["lower"])
        williams_r = np.nan_to_num(indicators["williams_r"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Lagged Williams %R for momentum filter
        williams_r_lagged = np.roll(williams_r, 1)
        
        # Entry conditions
        entry_condition = (
            (df["close"].values >= upper_band) &
            (williams_r < williams_r_oversold) &
            (williams_r > williams_r_lagged) &
            (df["close"].values >= upper_band + (1.5 * atr))
        )
        
        # Exit conditions - removed reference to non-existent entry_price column
        exit_condition = (
            (df["close"].values <= middle_band) |
            (williams_r > williams_r_overbought) |
            (df["close"].values <= df["close"].values - (stop_atr_mult * atr)) |
            (df["close"].values >= df["close"].values + (tp_atr_mult * atr))
        )
        
        # Generate signals
        entry_indices = np.where(entry_condition)[0]
        for i in entry_indices:
            if i >= warmup:
                signals.iloc[i] = 1.0  # LONG signal
                
        return signals