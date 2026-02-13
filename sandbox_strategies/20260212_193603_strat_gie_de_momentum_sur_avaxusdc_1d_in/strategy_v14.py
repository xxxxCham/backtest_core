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
        return ["rsi", "macd", "atr"]

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
        
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        warmup = int(params.get("warmup", 50))
        
        rsi = np.nan_to_num(indicators["rsi"])
        macd = indicators["macd"]
        macd_macd = np.nan_to_num(macd["macd"])
        macd_signal = np.nan_to_num(macd["signal"])
        atr = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)
        
        # Entry conditions
        rsi_long_condition = (rsi > rsi_oversold) & (np.roll(rsi, 1) <= rsi_oversold)
        macd_long_condition = (macd_macd > macd_signal) & (np.roll(macd_macd, 1) <= np.roll(macd_signal, 1))
        entry_long = rsi_long_condition & macd_long_condition
        
        rsi_short_condition = (rsi < rsi_overbought) & (np.roll(rsi, 1) >= rsi_overbought)
        macd_short_condition = (macd_macd < macd_signal) & (np.roll(macd_macd, 1) >= np.roll(macd_signal, 1))
        entry_short = rsi_short_condition & macd_short_condition
        
        # Exit conditions
        exit_long = (rsi > rsi_overbought) | (rsi < rsi_oversold) | ((rsi > rsi_overbought) & (close < np.roll(close, 1))) | ((rsi < rsi_oversold) & (close > np.roll(close, 1)))
        exit_short = (rsi > rsi_overbought) | (rsi < rsi_oversold) | ((rsi > rsi_overbought) & (close < np.roll(close, 1))) | ((rsi < rsi_oversold) & (close > np.roll(close, 1)))
        
        # Initialize positions
        position = 0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        
        # Generate signals
        for i in range(warmup, len(df)):
            if position == 0:
                if entry_long[i]:
                    position = 1
                    entry_price = close[i]
                    stop_loss = entry_price - (atr[i] * stop_atr_mult)
                    take_profit = entry_price + (atr[i] * tp_atr_mult)
                    signals.iloc[i] = 1.0
                elif entry_short[i]:
                    position = -1
                    entry_price = close[i]
                    stop_loss = entry_price + (atr[i] * stop_atr_mult)
                    take_profit = entry_price - (atr[i] * tp_atr_mult)
                    signals.iloc[i] = -1.0
            else:
                if position == 1:
                    if exit_long[i] or close[i] <= stop_loss or close[i] >= take_profit:
                        signals.iloc[i] = 0.0
                        position = 0
                elif position == -1:
                    if exit_short[i] or close[i] >= stop_loss or close[i] <= take_profit:
                        signals.iloc[i] = 0.0
                        position = 0
                        
        signals.iloc[:warmup] = 0.0
        return signals