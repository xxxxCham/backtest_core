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
        
        signals = pd.Series(np.zeros(len(df), dtype=np.float64))  # Initialize signal series with zeros
        n = len(df)  
        warmup = int(params.get('warmup', 50))   
        long_mask, short_mask = np.full(n, True), np.full(n,True)    
        
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        for i, row in df.iterrows():  
            # Remove unnecessary condition checking here
            
            if not any(indicators): 
                signals[i] += 1.0   
                
        
        signals = signals[:warmup].fillna(0) + (signals[warmup:] - signals[:warmup]).apply(lambda x: max(-x, 0)).cumsum()/abs(signals[:warmup])   # Calculate signal with warm up period
            
        return signals