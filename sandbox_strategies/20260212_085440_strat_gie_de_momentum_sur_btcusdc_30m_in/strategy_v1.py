from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="BTCUSDC_30m_Momentum_Strategy")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "macd", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 2.0, "tp_atr_mult": 4.5}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(10, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 4.0, 0.1),
            "tp_atr_mult": ParameterSpec(3.0, 7.0, 0.1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        rsi = np.nan_to_num(indicators["rsi"])
        macd = indicators["macd"]
        macd_hist = np.nan_to_num(macd["histogram"])
        atr = np.nan_to_num(indicators["atr"])
        
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        rsi_period = params["rsi_period"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        
        rsi_momentum = rsi - np.roll(rsi, 1)
        rsi_crossed_above_50 = (rsi > 50) & (np.roll(rsi, 1) <= 50)
        rsi_crossed_below_50 = (rsi < 50) & (np.roll(rsi, 1) >= 50)
        macd_positive = macd_hist > 0
        macd_negative = macd_hist < 0
        rsi_momentum_positive = rsi_momentum > 0
        rsi_momentum_negative = rsi_momentum < 0
        
        long_entry = (
            rsi_crossed_above_50 
            & macd_positive 
            & rsi_momentum_positive
        )
        
        long_exit = (
            rsi_crossed_below_50 
            | (rsi < rsi_oversold)
        )
        
        # Initialize entry and exit signals
        entry_long = np.zeros(len(df), dtype=bool)
        exit_long = np.zeros(len(df), dtype=bool)
        
        entry_long[1:] = long_entry[1:]
        exit_long[1:] = long_exit[1:]
        
        # Set initial signals
        positions = pd.Series(0.0, index=df.index)
        in_position = False
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        
        for i in range(1, len(df)):
            if not in_position and entry_long[i]:
                positions.iloc[i] = 1.0
                in_position = True
                entry_price = df["close"].iloc[i]
                stop_loss = entry_price - (stop_atr_mult * atr[i])
                take_profit = entry_price + (tp_atr_mult * atr[i])
            elif in_position:
                current_price = df["close"].iloc[i]
                if current_price <= stop_loss or current_price >= take_profit or exit_long[i]:
                    positions.iloc[i] = 0.0
                    in_position = False
                else:
                    positions.iloc[i] = 1.0
        
        signals = positions
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        return signals