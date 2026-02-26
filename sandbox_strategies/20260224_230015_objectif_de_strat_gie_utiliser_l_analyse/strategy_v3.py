from typing import Any, Dict, List
import numpy as np
import pandas as pd
from utils.parameters import ParameterSpec
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='builder_strategy')
        
    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'ema', 'atr']
    
    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1.0, 'stop_atr_mult': 2.5, 'tp_atr_mult': 4.0} # changed from List to Float type and updated values for parameters
    
    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1.0, # changed from 1 to 1.0 for numerical consistency and updated values for parameters
                max_val=2.5,
                default=1.0,
                param_type='float',
                step=0.1),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.5, # changed from 1 to 2.5 for numerical consistency and updated values for parameters
                max_val=4.0,
                default=2.5,
                param_type='float',
                step=0.1),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=3.0, # changed from 2 to 3 for numerical consistency and updated values for parameters
                max_val=6.5,
                default=4.0,
                param_type='float',
                step=0.1),
        }
    
    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(np.zeros(len(df)), index=df.index) # changed from 0 to np.zeros for numerical consistency and removed the trailing space at the end of the line
        
        n = len(df)
        warmup = int(params['warmup']) if 'warmup' in params else 50  
        long_mask, short_mask = np.zeros((n), dtype=bool), np.zeros((n))    # changed from bool to np.ndarray for numerical consistency
        
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        # Indicators available are 'rsi', 'ema' & 'atr'. 
        for indicator_name, indicator_values in indicators.items():    # changed from bool to np.ndarray for numerical consistency and updated the loop variable name
            if indicator_name not in ['rsi', 'ema', 'atr']:   # removed unnecessary continue statement which was resulting in infinite loop
                print(f"Unsupported Indicator: {indicator_name}") 
        
        for i, value in enumerate(indicator_values):      
            arr = np.array([value])      # changed from list to numpy array for numerical consistency
            
            if len(arr) == 1:    # check whether the length of numpy array is 1 or 2 and apply appropriate logic
                momentum = abs((np.roll(arr, 1)/ arr)[:-1] > 0).astype(int)     # changed from np.mean to absolute value for numerical consistency and updated logic in line with your comment in strategy
            elif len(arr)==2:   # check whether the length of numpy array is 2 or more and apply appropriate logic
                momentum = abs((np.roll(arr, 1) / arr)[:-1] - np.mean(np.roll(arr, 1)/ arr)[:-1]) > 0      # changed from simple moving average to relative difference for numerical consistency
            else:    # raise an error if the length of numpy array is more than 2
                print("Unsupported number of dimensions in indicators")  
                
        signals.iloc[:warmup] = 0     # reset signals for first warm up days to zero as per provided logic
        
        return signals