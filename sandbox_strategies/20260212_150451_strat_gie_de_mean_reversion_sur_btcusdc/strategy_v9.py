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
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.0, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(param_type="int", min_value=50, max_value=100, step=5),
            "rsi_oversold": ParameterSpec(param_type="int", min_value=0, max_value=50, step=5),
            "rsi_period": ParameterSpec(param_type="int", min_value=5, max_value=30, step=5),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=0.5, max_value=2.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=4.0, step=0.5),
            "warmup": ParameterSpec(param_type="int", min_value=20, max_value=100, step=10),
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
        
        donchian = indicators["donchian"]
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])
        
        donchian_upper = np.nan_to_num(donchian["upper"])
        donchian_lower = np.nan_to_num(donchian["lower"])
        donchian_middle = np.nan_to_num(donchian["middle"])
        
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.0)
        tp_atr_mult = params.get("tp_atr_mult", 2.0)
        
        # Entry condition: price touches lower Donchian band with oversold RSI
        entry_long = (df["close"] <= donchian_lower) & (rsi < rsi_oversold) & (df["close"] > donchian_middle)
        
        # Exit condition: price returns to Donchian middle
        exit_signal = df["close"] >= donchian_middle
        
        # Generate signals
        entry_indices = np.where(entry_long)[0]
        exit_indices = np.where(exit_signal)[0]
        
        for i in entry_indices:
            signals.iloc[i] = 1.0  # LONG signal
            
        # Set stop-loss and take-profit levels for long positions
        for i in range(len(signals)):
            if signals.iloc[i] == 1.0:
                # Stop loss at 1x ATR below entry price
                stop_loss = df["close"].iloc[i] - (atr[i] * stop_atr_mult)
                # Take profit at 2x ATR above entry price
                take_profit = df["close"].iloc[i] + (atr[i] * tp_atr_mult)
                
                # Check for exit conditions in subsequent bars
                for j in range(i+1, len(signals)):
                    current_price = df["close"].iloc[j]
                    if current_price <= stop_loss:
                        signals.iloc[j] = 0.0
                        break
                    elif current_price >= take_profit:
                        signals.iloc[j] = 0.0
                        break
                    elif current_price >= donchian_middle[i]:
                        signals.iloc[j] = 0.0
                        break
        
        return signals