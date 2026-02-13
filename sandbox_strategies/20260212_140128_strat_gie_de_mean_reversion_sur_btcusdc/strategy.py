from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mean_reversion_donchian_williamsr_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["donchian", "williams_r", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"donchian_period": 20, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50, "williams_r_period": 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "donchian_period": ParameterSpec(10, 50, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
            "warmup": ParameterSpec(30, 100, 1),
            "williams_r_period": ParameterSpec(10, 30, 1),
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
        williams_r_period = int(params.get("williams_r_period", 14))
        
        donchian = indicators["donchian"]
        williams_r = indicators["williams_r"]
        atr = indicators["atr"]
        
        donchian_upper = np.nan_to_num(donchian["upper"])
        donchian_middle = np.nan_to_num(donchian["middle"])
        donchian_lower = np.nan_to_num(donchian["lower"])
        williams_r_values = np.nan_to_num(williams_r)
        atr_values = np.nan_to_num(atr)
        
        # Entry condition: price touches lower band and Williams %R is oversold
        entry_condition = (df["close"].values == donchian_lower) & (williams_r_values < -80)
        
        # Exit condition: price reaches middle band or Williams %R shows overbought
        exit_condition = (df["close"].values >= donchian_middle) | (williams_r_values > -20)
        
        # Stop-loss and take-profit levels
        entry_prices = df["close"].values
        stop_loss = entry_prices - stop_atr_mult * atr_values
        take_profit = entry_prices + tp_atr_mult * atr_values
        
        # Initialize signals
        long_positions = np.zeros_like(df["close"], dtype=bool)
        in_position = False
        
        for i in range(len(df)):
            if not in_position and entry_condition[i]:
                long_positions[i] = True
                in_position = True
            elif in_position:
                # Check exit condition
                if exit_condition[i]:
                    long_positions[i] = False
                    in_position = False
                # Check stop-loss or take-profit
                elif df["close"].values[i] <= stop_loss[i] or df["close"].values[i] >= take_profit[i]:
                    long_positions[i] = False
                    in_position = False
        
        signals[long_positions] = 1.0
        return signals