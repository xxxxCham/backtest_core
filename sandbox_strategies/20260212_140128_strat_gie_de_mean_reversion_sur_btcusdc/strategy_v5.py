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
            "donchian_period": ParameterSpec(param_type="int", min_value=5, max_value=50, step=5),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=3.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=5.0, step=0.5),
            "warmup": ParameterSpec(param_type="int", min_value=30, max_value=100, step=10),
            "williams_r_overbought": ParameterSpec(param_type="int", min_value=-50, max_value=-10, step=5),
            "williams_r_oversold": ParameterSpec(param_type="int", min_value=-90, max_value=-70, step=5),
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
        williams_r_oversold = float(params.get("williams_r_oversold", -80))
        williams_r_period = int(params.get("williams_r_period", 14))
        
        donchian = indicators["donchian"]
        upper_band = np.nan_to_num(donchian["upper"])
        middle_band = np.nan_to_num(donchian["middle"])
        lower_band = np.nan_to_num(donchian["lower"])
        
        williams_r = np.nan_to_num(indicators["williams_r"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Lagged Williams %R for momentum filter
        williams_r_lagged = np.roll(williams_r, 1)
        
        # Entry condition: price touches upper band, Williams %R in oversold, and price is 1.5x ATR away from upper band
        entry_condition = (
            (np.nan_to_num(df["close"]) >= upper_band) &
            (williams_r < williams_r_oversold) &
            (williams_r > williams_r_lagged) &
            (np.nan_to_num(df["close"]) >= upper_band + (1.5 * atr))
        )
        
        # Long signals
        long_signal = pd.Series(0.0, index=df.index)
        long_signal[entry_condition] = 1.0
        
        # Exit conditions
        entry_prices = np.full(len(df), np.nan)
        entry_prices[entry_condition] = np.nan_to_num(df["close"])[entry_condition]
        
        # Stop-loss and take-profit levels
        stop_loss = entry_prices - (stop_atr_mult * atr)
        take_profit = entry_prices + (tp_atr_mult * atr)
        
        # Initialize exit conditions
        exit_condition = pd.Series(False, index=df.index)
        
        # Exit if price touches middle band
        exit_condition |= (np.nan_to_num(df["close"]) <= middle_band)
        
        # Exit if Williams %R enters overbought zone
        exit_condition |= (williams_r > williams_r_overbought)
        
        # Exit if stop-loss is hit
        exit_condition |= (np.nan_to_num(df["close"]) <= stop_loss)
        
        # Exit if take-profit is hit
        exit_condition |= (np.nan_to_num(df["close"]) >= take_profit)
        
        # Set signals to flat when exit condition is met
        signals[exit_condition] = 0.0
        signals[long_signal == 1.0] = 1.0
        
        return signals