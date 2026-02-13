from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="momentum_short_with_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "mfi", "atr", "ema"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(param_type="int", min_value=60, max_value=80, step=5),
            "rsi_oversold": ParameterSpec(param_type="int", min_value=10, max_value=30, step=5),
            "rsi_period": ParameterSpec(param_type="int", min_value=10, max_value=20, step=2),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=2.0, step=0.2),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=4.0, step=0.5),
            "warmup": ParameterSpec(param_type="int", min_value=30, max_value=70, step=10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        rsi = np.nan_to_num(indicators["rsi"])
        mfi = np.nan_to_num(indicators["mfi"])
        atr = np.nan_to_num(indicators["atr"])
        ema_20 = np.nan_to_num(indicators["ema"])
        close = np.nan_to_num(df["close"].values)
        
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        
        # Entry conditions for short
        rsi_condition = rsi > rsi_overbought
        mfi_condition = mfi > 50
        ema_condition = close > ema_20
        rsi_trend = rsi > np.roll(rsi, 1)
        
        short_entry = rsi_condition & mfi_condition & ema_condition & rsi_trend
        
        # Exit conditions
        # We'll use a simple approach: exit on RSI < oversold or ATR-based stops
        # For simplicity, we'll compute stop loss and take profit levels on entry
        
        entry_positions = np.where(short_entry)[0]
        exit_signals = np.zeros_like(short_entry, dtype=float)
        
        for i in entry_positions:
            entry_price = close[i]
            entry_atr = atr[i]
            stop_loss = entry_price + (entry_atr * stop_atr_mult)
            take_profit = entry_price - (entry_atr * tp_atr_mult)
            
            # Look ahead for exit conditions
            for j in range(i+1, len(close)):
                current_price = close[j]
                rsi_val = rsi[j]
                
                # Exit on stop loss
                if current_price >= stop_loss:
                    exit_signals[j] = 1.0  # FLAT
                    break
                
                # Exit on take profit
                if current_price <= take_profit:
                    exit_signals[j] = 1.0  # FLAT
                    break
                
                # Exit on RSI < oversold
                if rsi_val <= rsi_oversold:
                    exit_signals[j] = 1.0  # FLAT
                    break
        
        # Convert exit signals to short signals
        signals.iloc[entry_positions] = -1.0  # SHORT
        signals.iloc[exit_signals == 1.0] = 0.0  # FLAT
        
        return signals