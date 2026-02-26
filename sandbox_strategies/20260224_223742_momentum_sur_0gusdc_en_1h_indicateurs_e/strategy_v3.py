from typing import Any, Dict, List
import numpy as np
import pandas as pd
from utils.parameters import ParameterSpec
from strategies.base import StrategyBase
class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Snake_Case_Name')
        
    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'atr']
    
    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
                 'rsi_overbought': 70,
                 'rsi_oversold': 30,
                 'rsi_period': 14,
                 'stop_atr_mult': 1.5,
                 'tp_atr_mult': 3.0,
                 'warmup': 50}
    
    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(name='rsi_period', min_val=5, max_val=50, default=14, param_type='int'),
            'stop_atr_mult': ParameterSpec(name='stop_atr_mult', min_val=0.5, max_val=3.9, default=1.5, param_type='float'),
            'leverage': ParameterSpec(name='leverage', min_val=1, max_val=2, default=1, param_type='int'),
            'tp_atr_mult': ParameterSpec(name='tp_atr_mult', min_val=2.0, max_val=4.5, default=3.0, param_type='float')}
    
    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(np.zeros(len(df), dtype=np.float64))
        
        # Assuming that `indicators` dictionary has keys 'rsi' and 'atr'. 
        # If there are more indicators, you can extend the following for loop accordingly.
        rsi = indicators['rsi']
        atr = indicators['atr']
        
        n = len(df)
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        
        # Logic for generating signals based on RSI and ATR goes here...
        # ...
        
        return signals