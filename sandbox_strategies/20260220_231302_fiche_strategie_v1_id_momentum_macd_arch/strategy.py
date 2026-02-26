from typing import Any, Dict, List

import numpy as np
import pandas as pd

from strategies.base import StrategyBase
from utils.parameters import ParameterSpec

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='snake_case_name')
        
    @property
    def required_indicators(self) -> List[str]:
        return ['rsi']
    
    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'rsi_overbought ': 70, 'rsi_oversold ': 30, 'rsi_period ': 14, 'stop_atr_mult': 1.5, 'warmup': 50}
    
    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(min_val=5, max_val=50, default=14, param_type='int', step=None),
            'stop_atr_mult ': ParameterSpec(min_val=0.5, max_val=4.0, default=1.5, param_type='float', step=None),
            'leverage': ParameterSpec(min_val=1, max_val=2, default=1, param_type='int', step=1),
            'warmup': ParameterSpec(min_val=0, max_val=1000, default=50, param_type='int'),
        }
    
    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(np.zeros(len(df), dtype=np.float64), index=df.index)
        
        # Implement logic here to generate long and short signals based on rsi values
        n = len(df)
        warmup = params['warmup']
        long_mask = np.zeros((n, ), dtype=bool) 
        short_mask = np.zeros((n, ), dtype=bool)   
        
        # Add logic to generate signals here
        signals[:warmup] = 0  
      
        return signals