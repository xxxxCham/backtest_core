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
            "williams_r_period": ParameterSpec(param_type="int", min_value=5, max_value=30, step=5),
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
        
        # Wrap with np.nan_to_num for safe comparisons
        donchian_upper = np.nan_to_num(donchian["upper"])
        donchian_lower = np.nan_to_num(donchian["lower"])
        donchian_middle = np.nan_to_num(donchian["middle"])
        williams_r_val = np.nan_to_num(williams_r)
        atr_val = np.nan_to_num(atr)
        
        # Get params
        donchian_period = params.get("donchian_period", 20)
        williams_r_period = params.get("williams_r_period", 14)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        
        # Create entry conditions
        # Entry short: Price closes at or above upper Donchian band AND Williams %R < -20
        entry_short_condition = (df["close"] >= donchian_upper) & (williams_r_val < -20)
        
        # Exit condition: Close crosses back to middle Donchian band OR Williams %R continues in trend direction
        # For short: Exit if price crosses middle band downward or Williams %R continues in upward trend
        exit_short_condition = (df["close"] <= donchian_middle) | (williams_r_val > -20)
        
        # Generate signals
        entry_short = entry_short_condition
        exit_short = exit_short_condition
        
        # Initialize signal array
        short_signal = pd.Series(0.0, index=df.index)
        
        # Mark short entries
        short_signal[entry_short] = -1.0
        
        # Apply exit conditions
        # For simplicity, we'll assume that when we're in a short position,
        # we exit on the first signal that satisfies exit condition
        in_position = False
        position_entry_price = 0.0
        position_entry_atr = 0.0
        
        for i in range(len(short_signal)):
            if short_signal.iloc[i] == -1.0 and not in_position:
                # Enter short position
                in_position = True
                position_entry_price = df["close"].iloc[i]
                position_entry_atr = atr_val[i]
            elif in_position:
                # Check exit conditions
                if exit_short.iloc[i]:
                    # Exit short position
                    in_position = False
                    short_signal.iloc[i] = 0.0
                # Check stop loss
                elif (position_entry_price - df["close"].iloc[i]) > (position_entry_atr * stop_atr_mult):
                    # Stop loss triggered
                    in_position = False
                    short_signal.iloc[i] = 0.0
                # Check take profit
                elif (position_entry_price - df["close"].iloc[i]) > (position_entry_atr * tp_atr_mult):
                    # Take profit triggered
                    in_position = False
                    short_signal.iloc[i] = 0.0
        
        signals = short_signal
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals