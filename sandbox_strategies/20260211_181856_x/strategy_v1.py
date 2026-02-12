from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase
class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='x')
    @property
    def required_indicators(self):
        return ['rsi','bollinger']
    @property
    def default_params(self):
        return {'a':1}
    def generate_signals(self, df, indicators, params):
        s = pd.Series(0.0, index=df.index)
        r = np.nan_to_num(indicators['rsi'])
        bb = indicators['bollinger']
        lower = np.nan_to_num(bb['lower'])
        s[(r < 30) & (df['close'].values < lower)] = 1.0
        return s