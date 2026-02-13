from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="scalping_bollinger_vwap_atr_rsi")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "vwap", "atr", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 90, 1),
            "rsi_oversold": ParameterSpec(10, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        bb = indicators["bollinger"]
        vwap = np.nan_to_num(indicators["vwap"])
        atr = np.nan_to_num(indicators["atr"])
        rsi = np.nan_to_num(indicators["rsi"])
        
        close = np.nan_to_num(df["close"].values)
        open_ = np.nan_to_num(df["open"].values)
        
        bb_upper = np.nan_to_num(bb["upper"])
        bb_lower = np.nan_to_num(bb["lower"])
        
        # RSI params
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        
        # ATR params
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        
        # Warmup
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Entry conditions
        # Long entry: close crosses below bb_lower AND vwap < close AND rsi < oversold
        close_below_lower = close < bb_lower
        vwap_below_close = vwap < close
        rsi_below_oversold = rsi < rsi_oversold
        
        # Short entry: close crosses above bb_upper AND vwap > close AND rsi > overbought
        close_above_upper = close > bb_upper
        vwap_above_close = vwap > close
        rsi_above_overbought = rsi > rsi_overbought
        
        # Entry signals
        long_entry = close_below_lower & vwap_below_close & rsi_below_oversold
        short_entry = close_above_upper & vwap_above_close & rsi_above_overbought
        
        # Generate signals
        long_signal = pd.Series(0.0, index=df.index)
        short_signal = pd.Series(0.0, index=df.index)
        
        # Set entry signals
        long_signal[long_entry] = 1.0
        short_signal[short_entry] = -1.0
        
        # Combine signals
        signals = long_signal + short_signal
        
        return signals