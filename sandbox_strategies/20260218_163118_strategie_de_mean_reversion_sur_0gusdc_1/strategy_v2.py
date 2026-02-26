from typing import Any, Dict, List
import pandas as pd
import numpy as np
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
        
        if 'warmup' in params and params['warmup'] > 0:
            warmup_idx = df.index[:params['warmup']]
            
            signals[warmup_idx] = 1.0 # Set all values to 1 for the warm up period
        
        n = len(df)
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        
        rsi_val = indicators['rsi'] # Assuming 'rsi' is a pandas series
        ema_val = indicators['ema'] 
        atr_val = indicators['atr']
    
        # Implement your logic here to generate long/short signals based on RSI, EMA and ATR.
        
        return signals