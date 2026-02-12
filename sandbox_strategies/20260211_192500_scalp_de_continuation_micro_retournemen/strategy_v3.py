from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btc_ema_rsi_bollinger_scalper")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(param_type="int", min_value=60, max_value=90, step=5),
            "rsi_oversold": ParameterSpec(param_type="int", min_value=10, max_value=40, step=5),
            "rsi_period": ParameterSpec(param_type="int", min_value=10, max_value=20, step=2),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=3.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=5.0, step=0.5),
            "warmup": ParameterSpec(param_type="int", min_value=30, max_value=100, step=10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        ema9 = np.nan_to_num(indicators["ema"]["ema9"])
        ema21 = np.nan_to_num(indicators["ema"]["ema21"])
        ema50 = np.nan_to_num(indicators["ema"]["ema50"])
        rsi = np.nan_to_num(indicators["rsi"])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        atr = np.nan_to_num(indicators["atr"])
        
        # RSI parameters
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_period = params.get("rsi_period", 14)
        
        # Stop loss and take profit parameters
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        
        # Warmup
        warmup = int(params.get("warmup", 50))
        
        # Entry conditions
        # Long entry: price < ema21, rsi > oversold, rsi crosses above oversold, price > lower bollinger band, close > ema21
        long_condition = (
            (df["close"].values < ema21) &
            (rsi > rsi_oversold) &
            (np.roll(rsi, 1) <= rsi_oversold) &
            (df["close"].values > bb_lower) &
            (df["close"].values > ema21)
        )
        
        # Short entry: price > ema21, rsi < overbought, rsi crosses below overbought, price < upper bollinger band, close < ema21
        short_condition = (
            (df["close"].values > ema21) &
            (rsi < rsi_overbought) &
            (np.roll(rsi, 1) >= rsi_overbought) &
            (df["close"].values < bb_upper) &
            (df["close"].values < ema21)
        )
        
        # Exit conditions
        # Exit long if close crosses above upper bollinger band or rsi crosses below oversold and price < ema21
        long_exit = (
            (df["close"].values > bb_upper) |
            ((rsi < rsi_oversold) & (df["close"].values < ema21))
        )
        
        # Exit short if close crosses below lower bollinger band or rsi crosses above overbought and price > ema21
        short_exit = (
            (df["close"].values < bb_lower) |
            ((rsi > rsi_overbought) & (df["close"].values > ema21))
        )
        
        # Generate signals
        long_signals = np.zeros_like(df["close"], dtype=float)
        short_signals = np.zeros_like(df["close"], dtype=float)
        
        # Initialize positions
        long_positions = np.zeros_like(df["close"], dtype=bool)
        short_positions = np.zeros_like(df["close"], dtype=bool)
        
        # Loop through data to apply logic
        for i in range(1, len(df)):
            if long_condition[i]:
                long_signals[i] = 1.0
                long_positions[i] = True
            elif short_condition[i]:
                short_signals[i] = -1.0
                short_positions[i] = True
            elif long_positions[i-1] and long_exit[i]:
                long_signals[i] = 0.0
                long_positions[i] = False
            elif short_positions[i-1] and short_exit[i]:
                short_signals[i] = 0.0
                short_positions[i] = False
        
        # Combine signals
        signals = pd.Series(long_signals - short_signals, index=df.index, dtype=np.float64)
        
        # Ensure warmup is set
        signals.iloc[:warmup] = 0.0
        
        return signals