from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="supertrend_rsi_filter")

    @property
    def required_indicators(self) -> List[str]:
        return ["supertrend", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 60, "rsi_oversold": 40, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec("rsi_overbought", 50, 80, 1, 60),
            "rsi_oversold": ParameterSpec("rsi_oversold", 20, 50, 1, 40),
            "rsi_period": ParameterSpec("rsi_period", 5, 30, 1, 14),
            "stop_atr_mult": ParameterSpec("stop_atr_mult", 1.0, 3.0, 0.1, 1.5),
            "tp_atr_mult": ParameterSpec("tp_atr_mult", 2.0, 5.0, 0.1, 3.0),
            "warmup": ParameterSpec("warmup", 20, 100, 1, 50),
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
        
        # Extract indicators
        supertrend = indicators["supertrend"]
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)
        
        # Supertrend bands
        upper_band = np.nan_to_num(supertrend["upper"])
        lower_band = np.nan_to_num(supertrend["lower"])
        
        # Entry conditions
        rsi_overbought = params.get("rsi_overbought", 60)
        rsi_oversold = params.get("rsi_oversold", 40)
        
        # Entry long: price crosses above upper band AND RSI is below oversold
        long_condition = (close > upper_band) & (rsi < rsi_oversold)
        
        # Entry short: price crosses below lower band AND RSI is above overbought
        short_condition = (close < lower_band) & (rsi > rsi_overbought)
        
        # Exit conditions
        # Exit long: price crosses below lower band
        exit_long_condition = close < lower_band
        
        # Exit short: price crosses above upper band
        exit_short_condition = close > upper_band
        
        # Initialize positions
        position = 0
        position_change = 0
        
        for i in range(warmup, len(signals)):
            if position == 0:
                if long_condition[i]:
                    signals[i] = 1.0
                    position = 1
                elif short_condition[i]:
                    signals[i] = -1.0
                    position = -1
            elif position == 1:
                if exit_long_condition[i]:
                    signals[i] = 0.0
                    position = 0
            elif position == -1:
                if exit_short_condition[i]:
                    signals[i] = 0.0
                    position = 0
                    
        return signals