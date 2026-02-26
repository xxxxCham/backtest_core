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
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 50}
    
    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=1,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=2.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
        }
    
    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        
        # Implement explicit LONG / SHORT / FLAT logic
        warmup = params['warmup']  # get the value of 'warmup' from default_params
        signals.iloc[:warmup] = 0.0  
        
        # Adding Bollinger Bands indicators as per requirement
        for indicator in ['rsi', 'ema', 'atr']:
            if indicator == 'bollinger':
                bollinger = pd.DataFrame()
                
                # Calculate upper band, middle band and lower band
                bands = df[indicator].rolling(window=20).mean().add(df[indicator].rolling(window=20).std(), 
                                                                     rsuffix='_dev').stack().reset_index()
                
                # Set the indicator column as index
                bollinger['level'] = bands.iloc[:, -1]  
                
            df[f'{indicator}_band'] = pd.Series(np.nan, index=df.index)  # Create a new series with NaN values for all elements
            
        signals[:warmup] = 0.0    # Reset signals to zero for the first 'warmup' observations
        
        return signals