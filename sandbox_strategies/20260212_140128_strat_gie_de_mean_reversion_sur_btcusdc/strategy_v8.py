from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="Mean-reversion_BTCUSDC_30m")

    @property
    def required_indicators(self) -> List[str]:
        return ["donchian", "williams_r", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"donchian_period": 20, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "williams_r_period": 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "donchian_period": ParameterSpec(param_type="int", min_value=10, max_value=50, step=5),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=3.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=5.0, step=0.5),
            "williams_r_period": ParameterSpec(param_type="int", min_value=10, max_value=30, step=5),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        donchian_period = params["donchian_period"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        williams_r_period = params["williams_r_period"]
        
        close = np.nan_to_num(df["close"].values)
        donchian = indicators["donchian"]
        upper_band = np.nan_to_num(donchian["upper"])
        lower_band = np.nan_to_num(donchian["lower"])
        williams_r = np.nan_to_num(indicators["williams_r"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Entry conditions for long
        entry_long = (close == lower_band) & (williams_r < -80)
        
        # Exit conditions for long
        exit_long = (close == upper_band) | (williams_r > -20)
        
        # Stop loss and take profit levels
        stop_loss = close - (stop_atr_mult * atr)
        take_profit = close + (tp_atr_mult * atr)
        
        # Initialize entry and exit signals
        long_entry = pd.Series(0.0, index=df.index)
        long_exit = pd.Series(0.0, index=df.index)
        
        # Find entry points
        entry_indices = np.where(entry_long)[0]
        for i in entry_indices:
            long_entry.iloc[i] = 1.0
            
        # Find exit points
        exit_indices = np.where(exit_long)[0]
        for i in exit_indices:
            long_exit.iloc[i] = -1.0
            
        # Combine signals
        signals = long_entry + long_exit
        signals = signals.replace(0.0, 0.0).replace(1.0, 1.0).replace(-1.0, -1.0)
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals