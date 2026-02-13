from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="Mean-Reversion_ETHUSDC_15m")

    @property
    def required_indicators(self) -> List[str]:
        return ["donchian", "williams_r", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"atr_period": 14, "donchian_period": 20, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "williams_r_period": 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "atr_period": ParameterSpec(param_type="int", min_value=5, max_value=30, step=5),
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
        
        # Extract indicators
        donchian = indicators["donchian"]
        williams_r = indicators["williams_r"]
        atr = indicators["atr"]
        
        # Get donchian values
        upper_band = np.nan_to_num(donchian["upper"])
        lower_band = np.nan_to_num(donchian["lower"])
        middle_band = np.nan_to_num(donchian["middle"])
        
        # Get Williams %R values
        williams_r_val = np.nan_to_num(williams_r)
        
        # Get ATR values
        atr_val = np.nan_to_num(atr)
        
        # Get close prices
        close = np.nan_to_num(df["close"].values)
        
        # Entry conditions for short
        entry_short = (close == upper_band) & (williams_r_val < -20)
        
        # Exit conditions
        exit_short = (close == middle_band) | (williams_r_val > -20)
        
        # Generate signals
        short_entries = np.where(entry_short, -1.0, 0.0)
        short_exits = np.where(exit_short, 0.0, -1.0)
        
        # Combine signals
        signals = pd.Series(short_entries, index=df.index, dtype=np.float64)
        
        # Apply exits where needed
        for i in range(len(signals)):
            if signals.iloc[i] == -1.0:
                # Check if we should exit due to Williams %R continuing trend
                if i < len(signals) - 1 and williams_r_val[i+1] > -20:
                    signals.iloc[i] = 0.0
                # Check if we should exit due to crossing middle band
                elif i < len(signals) - 1 and close[i+1] == middle_band:
                    signals.iloc[i] = 0.0
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals