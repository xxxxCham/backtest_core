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
        return {"donchian_period": 20, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50, "williams_r_overbought": -20, "williams_r_oversold": -80, "williams_r_period": 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "donchian_period": ParameterSpec("donchian_period", 10, 50, 1),
            "stop_atr_mult": ParameterSpec("stop_atr_mult", 1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec("tp_atr_mult", 2.0, 5.0, 0.1),
            "williams_r_overbought": ParameterSpec("williams_r_overbought", -30, -10, 1),
            "williams_r_oversold": ParameterSpec("williams_r_oversold", -90, -70, 1),
            "williams_r_period": ParameterSpec("williams_r_period", 10, 30, 1),
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
        williams_r_period = int(params.get("williams_r_period", 14))
        
        donchian = indicators["donchian"]
        upper_band = np.nan_to_num(donchian["upper"])
        middle_band = np.nan_to_num(donchian["middle"])
        lower_band = np.nan_to_num(donchian["lower"])
        
        williams_r = np.nan_to_num(indicators["williams_r"])
        williams_r_lagged = np.roll(williams_r, 1)
        williams_r_lagged[0] = 0
        
        atr = np.nan_to_num(indicators["atr"])
        
        # Entry condition: close touches upper band, Williams %R in oversold, and trending up
        entry_condition = (df["close"].values >= upper_band * 0.99) & (williams_r < williams_r_oversold) & (williams_r > williams_r_lagged)
        
        # Exit condition: close touches middle band or Williams %R returns to overbought
        exit_condition = (df["close"].values <= middle_band) | (williams_r > williams_r_overbought)
        
        # Generate signals
        entry_points = np.where(entry_condition, 1.0, 0.0)
        exit_points = np.where(exit_condition, -1.0, 0.0)
        
        # Simple logic: enter on signal, exit when condition met
        signal_values = np.zeros_like(entry_points)
        in_position = False
        position_entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        
        for i in range(len(signal_values)):
            if not in_position and entry_points[i] == 1.0:
                signal_values[i] = 1.0
                in_position = True
                position_entry_price = df["close"].iloc[i]
                stop_loss = position_entry_price - (atr[i] * stop_atr_mult)
                take_profit = position_entry_price + (atr[i] * tp_atr_mult)
            elif in_position:
                if df["close"].iloc[i] <= stop_loss or df["close"].iloc[i] >= take_profit or exit_points[i] == -1.0:
                    signal_values[i] = -1.0
                    in_position = False
                else:
                    signal_values[i] = 1.0
            else:
                signal_values[i] = 0.0
                
        signals = pd.Series(signal_values, index=df.index, dtype=np.float64)
        return signals