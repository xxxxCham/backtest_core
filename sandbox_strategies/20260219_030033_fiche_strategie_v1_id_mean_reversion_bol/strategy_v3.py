from utils.parameters import ParameterSpec
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_bollinger_rsi')
    
    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'rsi', 'atr']
    
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
            'ATR stop/take-profit with concrete multipliers': ParameterSpec(
                name='ATR stop/take-profit with concrete multipliers',
                min_val=0.5,
                max_val=4.0,
                default=1.25,
                param_type='float',
                step=None,  # Removed 'step' from ParameterSpec dictionary as it is not necessary here.
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=0.5,
                max_val=6.0,
                default=5.5,
                param_type='float',
                step=None,  # Removed 'step' from ParameterSpec dictionary as it is not necessary here.
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=1,
                param_type='int',
                step=None,  # Removed 'step' from ParameterSpec dictionary as it is not necessary here.
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=2.0,
                default=1.5,
                param_type='float',
                step=None,  # Removed 'step' from ParameterSpec dictionary as it is not necessary here.
            ),
        }
    
    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
    
        # Create long_mask and short_mask using the correct property of Bollinger Bands (close < bollinger.lower) and RSI (rsi < 30).
        # But note that you need to define 'long_mask' and 'short_mask' before using them as they are boolean mask expressions.
        
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
    
        # Implement logic for generating signals here...
        
        return signals  # Return only Python code in one block