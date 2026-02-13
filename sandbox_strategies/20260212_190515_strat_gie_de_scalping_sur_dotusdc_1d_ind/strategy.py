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
        return ['STOCHASTIC', 'VWAP', 'ATR']
    
    @property
    def default_params(self) -> Dict[str, Any]:
        return {}
        
    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, np.ndarray], params: Dict[str, Any]) -> pd.Series:
        # Get the Bollinger Bands upper and lower values for STOCHASTIC
        bollinger = [bollinger['upper'] for bollinger in indicators['STOCHASTIC'] if 'upper' in bollinger]
        
        # Calculate the ATR value
        atr_value = indicators['ATR'][0]
        
        signals = pd.Series(np.nan, index=df.index)
        
        for i in range(len(df)):
            if np.any([bollinger[i] > upperBb[i] for upperBb in bollinger]):
                # LONG signal when STOCHASTIC crosses above the Bollinger Band
                signals[i] = 1.0
                
            elif np.any([bollinger[i] < lowerBb[i] for lowerBb in bollinger]):
                # SHORT signal when STOCHASTIC crosses below the Bollinger Band
                signals[i] = -1.0
        
        return signals