from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase
from utils.parameters import ParameterSpec

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_donchian_adx')
        
    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'donchian', 'atr']
    
    @property
    def default_params(self) -> Dict[str, Any]:
        return {'fees': 10,
                 'leverage': 1,
                 'slippage': 5,
                 'stop_atr_mult': 1.5,
                 'tp_atr_mult': 3.0,
                 'warmup': 50}
    
    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.8,
                max_val=4.0,
                default=2.25,
                param_type='float',
                step=None,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.5,
                max_val=6.0,
                default=3.5,
                param_type='float',
                step=None,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=1,
                param_type='int',
                step=1,
            ),
        }
    
    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
         warmup = int(params.get('warmup', 50))
         signals = pd.Series(0.0, index=df.index, dtype=np.float64)
         
         for ind in indicators: # iterate over each indicator in the dict
             if 'middle' in ind: 
                 middle_band = indicators[ind]['middle']   
                 close = df['close'].values   # assuming df is a DataFrame with a column named 'close'
                 
                 long_mask |= (df['close'] > np.roll(close, -1)) & ((adx > 25) | (params.get('stop_atr_mult', 0.8) < adx)) # bitwise AND and assignment in one line
                 short_mask &= ~(np.roll(close, -1).astype(bool) <= np.roll(indicators[ind]['lower'], -1)) & ((adx > 25) | (params.get('stop_atr_mult', 0.8) < adx)) # bitwise AND and assignment in one line
             elif 'upper' in ind:  
                 upper_band = indicators[ind]['upper']   
                 
                 long_mask |= (df['close'] > np.roll(close, -1)) & ((adx > 25) | (params.get('stop_atr_mult', 0.8) < adx)) # bitwise AND and assignment in one line
                 short_mask &= ~(np.roll(close, -1).astype(bool) <= np.roll(indicators[ind]['lower'], -1)) & ((adx > 25) | (params.get('stop_atr_mult', 0.8) < adx)) # bitwise AND and assignment in one line
             elif 'lower' in ind:  
                 lower_band = indicators[ind]['lower']   
                 
                 long_mask |= (np.roll(close, -1).astype(bool) >= np.roll(indicators[ind]['upper'], 1)) & ((adx > 25) | (params.get('stop_atr_mult', 0.8) < adx)) # bitwise AND and assignment in one line
                 short_mask &= ~np.roll(close, -1).astype(bool) <= np.roll(indicators[ind]['middle'], 1) & ((adx > 25) | (params.get('stop_atr_mult', 0.8) < adx)) # bitwise AND and assignment in one line
         
         signals[:warmup] = 0    # set initial values for the first warm-up period to zero  
     
         return signals