from typing import Dict

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="MicroRetournementBTC30m")
    
    @property
    def required_indicators(self) -> list:
        return ["ema", "rsi", "bollinger", "atr"]
    
    @property
    def default_params(self) -> dict:
        return {
            "atr_period": 14,
            "bb_period": 20,
            "bb_stddev": 2,
            "ema_period": 21,
            "rsi_period": 14,
            "risk_percent": 1
        }
    
    @property
    def parameter_specs(self) -> dict:
        return {
            "atr_period": ParameterSpec(name="atr_period", default=14, description="ATR period"),
            "bb_period": ParameterSpec(name="bb_period", default=20, description="Bollinger period"),
            "bb_stddev": ParameterSpec(name="bb_stddev", default=2, description="Bollinger std dev"),
            "ema_period": ParameterSpec(name="ema_period", default=21, description="EMA period"),
            "rsi_period": ParameterSpec(name="rsi_period", default=14, description="RSI period"),
            "risk_percent": ParameterSpec(name="risk_percent", default=1, description="Risk percent per trade")
        }
    
    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        n = len(df)
        signals = np.zeros(n, dtype=np.float64)
        
        close = df["close"]
        high = df["high"]
        low = df["low"]
        
        ema_val = np.nan_to_num(indicators["ema"])
        rsi_val = np.nan_to_num(indicators["rsi"])
        atr_val = np.nan_to_num(indicators["atr"])
        bb = indicators["bollinger"]
        upper = np.nan_to_num(bb["upper"])
        lower = np.nan_to_num(bb["lower"])
        
        for i in range(1, n):
            # Long: pullback to EMA21, RSI crossing up through 50, rejection of lower band
            if (rsi_val[i] > 50 and rsi_val[i-1] <= 50) and \
               close.iloc[i] > ema_val[i] and \
               close.iloc[i] > lower[i] and \
               close.iloc[i-1] <= lower[i-1]:
                signals[i] = 1.0
            # Short: spike to upper band, RSI crossing down from overbought, reversal below upper band but above EMA21
            elif (rsi_val[i] < 70 and rsi_val[i-1] >= 70) and \
                 high.iloc[i] >= upper[i] and \
                 close.iloc[i] < upper[i] and close.iloc[i] > ema_val[i]:
                signals[i] = -1.0
        
        return pd.Series(signals, index=df.index)