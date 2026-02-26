from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase
from utils.parameters import ParameterSpec

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ATR based Snake Case Name')
        
    @property
    def required_indicators(self) -> List[str]:
        return ['adx', 'atr', 'donchian', 'ema', 'ichimoku', 'keltner', 'macd', 'momentum', 'obv', 'aroon', 'supertrend', 'fear_greed', 'stochastic', 'vortex']
    
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
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=None,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
                param_type='float',
                step=None,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=1,
                param_type='int',
                step=0,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=3.0,
                param_type='float',
                step=None,
            ),
        }
    
    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Define your trading logic here
    
    return signals