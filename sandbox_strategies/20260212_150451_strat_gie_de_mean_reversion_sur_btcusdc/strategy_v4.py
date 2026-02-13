from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="donchian_rsi_mean_reversion")

    @property
    def required_indicators(self) -> List[str]:
        return ["donchian", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(50, 90, 1),
            "rsi_oversold": ParameterSpec(10, 50, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(1.5, 5.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1),
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
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Extract Donchian bands
        upper_band = np.nan_to_num(donchian["upper"])
        middle_band = np.nan_to_num(donchian["middle"])
        lower_band = np.nan_to_num(donchian["lower"])
        
        # Entry conditions
        rsi_overbought = params["rsi_overbought"]
        price = np.nan_to_num(df["close"].values)
        
        # Entry: price touches upper Donchian band AND RSI > 70
        entry_condition = (np.abs(price - upper_band) < 1e-8) & (rsi > rsi_overbought)
        
        # Exit: price crosses below middle Donchian band
        exit_condition = price < middle_band
        
        # Initialize entry and exit signals
        entry_signal = pd.Series(0.0, index=df.index)
        exit_signal = pd.Series(0.0, index=df.index)
        
        # Set entry signals
        entry_signal[entry_condition] = 1.0
        
        # Set exit signals
        exit_signal[exit_condition] = -1.0
        
        # Combine entry and exit signals
        # For simplicity, we'll use a basic approach: when entry is triggered, hold until exit
        # We'll set the signal to 1.0 when entry happens, and -1.0 when exit happens
        # But we need to avoid conflicting signals
        
        # Create a cumulative signal
        positions = pd.Series(0.0, index=df.index)
        
        # Initialize position tracking
        in_position = False
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        
        # Loop through each bar
        for i in range(len(df)):
            if not in_position and entry_signal.iloc[i] == 1.0:
                # Enter long
                in_position = True
                entry_price = price[i]
                stop_loss = entry_price - (atr[i] * params["stop_atr_mult"])
                take_profit = entry_price + (atr[i] * params["tp_atr_mult"])
                positions.iloc[i] = 1.0
            elif in_position:
                # Check exit conditions
                if price[i] <= stop_loss or price[i] >= take_profit:
                    # Exit
                    in_position = False
                    positions.iloc[i] = 0.0
                else:
                    # Still in position
                    positions.iloc[i] = 1.0
            else:
                # Not in position
                positions.iloc[i] = 0.0
                
        signals = positions
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals