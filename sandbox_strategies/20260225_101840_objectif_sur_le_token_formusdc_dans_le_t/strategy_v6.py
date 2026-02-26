from utils.parameters import ParameterSpec
from strategies.base import StrategyBase
from typing import Any, Dict, List
import numpy as np
import pandas as pd
# ... existing imports ... 

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='FormusdcMeanReversion')
        
    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'obv', 'bollinger']
    
    # ... existing properties ... 
    
    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # ... existing code ... 
    
        if len(crossover_signal) > 0 or len(short_crossover_signal) > 0 or \
           len(sell_signal) > 0 or len(buy_signal) > 0:
            signals[row['datetime']] = crossover_signal + short_crossover_signal + buy_signal + sell_signal
            
        # ... existing code ...
        
    return signals