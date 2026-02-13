from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="momentum_avaxusdc_1d")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(20, 40, 1),
            "rsi_period": ParameterSpec(10, 20, 1),
            "stop_atr_mult": ParameterSpec(1.0, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 4.0, 0.1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        rsi = np.nan_to_num(indicators["rsi"])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        atr = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)
        macd_histogram = np.nan_to_num(indicators["macd"]["histogram"])
        
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        warmup = int(params.get("warmup", 50))
        
        signals.iloc[:warmup] = 0.0
        
        # Entry long conditions
        entry_long_cond1 = (rsi > rsi_oversold) & (np.roll(rsi, 1) <= rsi_oversold)
        entry_long_cond2 = close > bb_upper
        entry_long_cond3 = macd_histogram > 0
        
        long_entries = entry_long_cond1 & entry_long_cond2 & entry_long_cond3
        
        # Entry short conditions
        entry_short_cond1 = (rsi < rsi_overbought) & (np.roll(rsi, 1) >= rsi_overbought)
        entry_short_cond2 = close < bb_lower
        entry_short_cond3 = macd_histogram < 0
        
        short_entries = entry_short_cond1 & entry_short_cond2 & entry_short_cond3
        
        # Exit conditions
        exit_long_cond1 = (rsi > rsi_overbought) | (rsi < rsi_oversold)
        exit_long_cond2 = (rsi < 50) & (close < np.roll(close, 1))
        exit_long_cond3 = (rsi > 50) & (close > np.roll(close, 1))
        
        exits = exit_long_cond1 | exit_long_cond2 | exit_long_cond3
        
        # Generate signals
        long_signals = pd.Series(0.0, index=df.index)
        short_signals = pd.Series(0.0, index=df.index)
        
        # Initialize positions
        position = 0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        
        for i in range(len(df)):
            if position == 0:
                if long_entries[i]:
                    position = 1
                    entry_price = close[i]
                    stop_loss = entry_price - (stop_atr_mult * atr[i])
                    take_profit = entry_price + (tp_atr_mult * atr[i])
                    long_signals.iloc[i] = 1.0
                elif short_entries[i]:
                    position = -1
                    entry_price = close[i]
                    stop_loss = entry_price + (stop_atr_mult * atr[i])
                    take_profit = entry_price - (tp_atr_mult * atr[i])
                    short_signals.iloc[i] = -1.0
            else:
                if position == 1:
                    if exits[i] or close[i] <= stop_loss or close[i] >= take_profit:
                        position = 0
                        long_signals.iloc[i] = 0.0
                elif position == -1:
                    if exits[i] or close[i] >= stop_loss or close[i] <= take_profit:
                        position = 0
                        short_signals.iloc[i] = 0.0
        
        signals = long_signals + short_signals
        
        return signals