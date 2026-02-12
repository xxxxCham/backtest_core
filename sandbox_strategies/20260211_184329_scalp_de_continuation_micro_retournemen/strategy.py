from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'bollinger_upper', 'bollinger_lower']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {}

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, pd.Series], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index)
        
        for i in range(len(df)):
            rsi = indicators['rsi'].iloc[i]
            boll_upper = indicators['bollinger_upper'].iloc[i]
            boll_lower = indicators['bollinger_lower'].iloc[i]
            close_price = df.iloc[i]['close']
            
            if rsi < 30 and close_price < boll_lower:
                signals.iloc[i] = 1.0
            elif rsi > 70 and close_price > boll_upper:
                signals.iloc[i] = -1.0
        
        return signals