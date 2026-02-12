from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")

    @property
    def required_indicators(self):
        return ['ema', 'rsi']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params) -> pd.Series:
        signals = pd.Series(0.0, index=df.index)
        
        ema_short = indicators['ema'][1]
        ema_long = indicators['ema'][2]
        rsi = indicators['rsi'][1]

        conditions = [
            (df.close > ema_short) & (df.close < df.local_max),
            (df.close > ema_short) & (df.close < ema_long),
            (rsi > 80)
        ]
        action = ['buy', 'buy', 'sell']

        for i, cond in enumerate(conditions):
            signals.loc[(cond), :] = action[i]

        return signals