from utils.parameters import ParameterSpec
from strategies.base import StrategyBase
from typing import Any, Dict, List
import numpy as np
import pandas as pd

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='builder_strategy')
        
    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'ema', 'atr']
    
    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 50}
        
    @property
    def parameter_specs(self) -> Dict[str, Any]:
        return {
            'leverage': {'min': 1, 'max': 2, 'default': 1, 'type': 'int', 'step': 1},
            'stop_atr_mult': {'min': 1.0, 'max': 2.0, 'default': 1.5, 'type': 'float', 'step': 0.1},
            'tp_atr_mult': {'min': 2.0, 'max': 4.5, 'default': 3.0, 'type': 'float', 'step': 0.1}
        }
    
    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Define long_mask and short_mask here 
        long_mask = np.zeros(len(df), dtype=bool)
        short_mask = np.zeros(len(df), dtype=bool)
    
        rsi, ema, atr = indicators['rsi'], indicators['ema'], indicators['atr']
        
        # Rest of your logic here...
        return signals