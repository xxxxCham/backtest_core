from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="momentum_btcusdc_30m_revised")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr", "macd"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(20, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        bb_upper = np.nan_to_num(bb["upper"])
        bb_lower = np.nan_to_num(bb["lower"])
        atr = np.nan_to_num(indicators["atr"])
        macd = indicators["macd"]
        macd_histogram = np.nan_to_num(macd["histogram"])
        
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        close = np.nan_to_num(df["close"].values)
        rsi_prev = np.roll(rsi, 1)
        rsi_prev[0] = rsi[0]
        
        # Entry conditions
        long_condition = (
            (rsi > rsi_overbought) &
            (macd_histogram > 0) &
            (rsi_prev <= rsi_overbought) &
            (close > bb_upper) &
            (rsi > 50)
        )
        
        short_condition = (
            (rsi < rsi_oversold) &
            (macd_histogram < 0) &
            (rsi_prev >= rsi_oversold) &
            (close < bb_lower) &
            (rsi < 50)
        )
        
        # Exit conditions
        exit_long = (
            (rsi > rsi_overbought + 5) |
            (rsi < rsi_oversold - 5) |
            (rsi > 50) & (rsi_prev <= 50)
        )
        
        exit_short = (
            (rsi < rsi_oversold - 5) |
            (rsi > rsi_overbought + 5) |
            (rsi < 50) & (rsi_prev >= 50)
        )
        
        # Initialize positions
        position = 0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        
        for i in range(warmup, len(df)):
            if position == 0:
                if long_condition[i]:
                    signals[i] = 1.0
                    position = 1
                    entry_price = close[i]
                    stop_loss = entry_price - (stop_atr_mult * atr[i])
                    take_profit = entry_price + (tp_atr_mult * atr[i])
                elif short_condition[i]:
                    signals[i] = -1.0
                    position = -1
                    entry_price = close[i]
                    stop_loss = entry_price + (stop_atr_mult * atr[i])
                    take_profit = entry_price - (tp_atr_mult * atr[i])
            elif position == 1:
                if exit_long[i] or close[i] <= stop_loss or close[i] >= take_profit:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 1.0
            elif position == -1:
                if exit_short[i] or close[i] >= stop_loss or close[i] <= take_profit:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -1.0
        
        return signals