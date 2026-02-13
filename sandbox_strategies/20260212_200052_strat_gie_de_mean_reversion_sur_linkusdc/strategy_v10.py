from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_trend_filter")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "rsi", "atr", "sma"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"bollinger_period": 20, "bollinger_std_dev": 2, "rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "sma_period": 50, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "bollinger_period": ParameterSpec(type=int, min_value=10, max_value=50, step=1),
            "bollinger_std_dev": ParameterSpec(type=float, min_value=1.0, max_value=3.0, step=0.1),
            "rsi_overbought": ParameterSpec(type=float, min_value=60, max_value=80, step=1),
            "rsi_oversold": ParameterSpec(type=float, min_value=20, max_value=40, step=1),
            "rsi_period": ParameterSpec(type=int, min_value=7, max_value=21, step=1),
            "sma_period": ParameterSpec(type=int, min_value=30, max_value=100, step=5),
            "stop_atr_mult": ParameterSpec(type=float, min_value=1.0, max_value=3.0, step=0.1),
            "tp_atr_mult": ParameterSpec(type=float, min_value=2.0, max_value=5.0, step=0.1),
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
        
        close = df['close'].values
        bb = indicators["bollinger"]
        bb_upper = np.nan_to_num(bb["upper"])
        bb_middle = np.nan_to_num(bb["middle"])
        bb_lower = np.nan_to_num(bb["lower"])
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])
        sma = np.nan_to_num(indicators["sma"])
        
        rsi_oversold = params["rsi_oversold"]
        rsi_overbought = params["rsi_overbought"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        
        long_entry = (close <= bb_lower) & (rsi < rsi_oversold) & (close > sma)
        short_entry = (close >= bb_upper) & (rsi > rsi_overbought) & (close < sma)
        
        long_exit = close >= bb_middle
        short_exit = close <= bb_middle
        
        position = 0
        for i in range(warmup, len(signals)):
            if position == 0:
                if long_entry[i]:
                    position = 1
                    entry_price = close[i]
                    stop_loss = entry_price - stop_atr_mult * atr[i]
                    take_profit = entry_price + tp_atr_mult * atr[i]
                elif short_entry[i]:
                    position = -1
                    entry_price = close[i]
                    stop_loss = entry_price + stop_atr_mult * atr[i]
                    take_profit = entry_price - tp_atr_mult * atr[i]
            elif position == 1:
                if close[i] <= stop_loss or close[i] >= take_profit or long_exit[i]:
                    position = 0
            elif position == -1:
                if close[i] >= stop_loss or close[i] <= take_profit or short_exit[i]:
                    position = 0
            
            signals.iloc[i] = float(position)
        
        return signals