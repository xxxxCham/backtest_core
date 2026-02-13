from utils.parameters import ParameterSpec
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")

    @property
    def required_indicators(self) -> List[str]:
        return ['keltner', 'supertrend', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {}

    def generate_signals(self, df: pd.DataFrame, indicators: dict, params: dict):
        keltner = indicators['keltner']['middle']  # use middle Bollinger band line (14 periods)
        supertrend = indicators['supertrend']['direction'] == 'UP'
        atr_values = indicators['atr'][0]  # take the first value for simplicity
        
        # Long strategy if Keltner Channel is up and Supertrend is UP, else short.
        signals = pd.Series(np.where((keltner > np.roll(keltner, 1)) & supertrend, 1., np.nan), index=df.index)
        
        return signals