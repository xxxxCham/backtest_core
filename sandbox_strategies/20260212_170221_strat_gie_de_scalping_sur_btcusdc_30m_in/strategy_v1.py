from utils.parameters import ParameterSpec
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="macd_rsi_scalping")
    
    @property
    def required_indicators(self) -> List[str]:
        return ["macd", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"atr_period": 14, "macd_fastperiod": 12, "macd_signalperiod": 9, "macd_slowperiod": 26, "overbought_level": 70, "oversold_level": 30, "rsi_period": 14, "stop_multiplier": 1.0, "take_profit_multiplier": 1.5}
    
    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        macd = np.nan_to_num(indicators["macd"]["macd"])
        signal = np.nan_to_num(indicators["macd"]["signal"])
        rsi = np.nan_to_num(indicators["rsi"])
        
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        for i in range(warmup, len(df)):
            if rsi[i-1] <= params["oversold_level"] and rsi[i] > params["oversold_level"]: # oversold to not oversold cross
                if macd[i-1] < signal[i-1] and macd[i] >= signal[i]: # MACD cross above signal line
                    signals.iloc[i] = 1.0 # LONG
            elif rsi[i-1] >= params["overbought_level"] and rsi[i] < params["overbought_level"]: # overbought to not overbought cross
                if macd[i-1] > signal[i-1] and macd[i] <= signal[i]: # MACD cross below signal line
                    signals.iloc[i] = -1.0 # SHORT
        
        return signals