from typing import Any, Dict, List
import numpy as np
import pandas as pd

from strategies.base import StrategyBase
from utils.parameters import ParameterSpec

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='FICHE_STRATEGIE')
        
    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'rsi', 'atr']
    
    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'stop_atr_mult': 2.5,
                'tp_atr_mult': 4.0,
                'warmup': 50}
    
    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ATR stop/take-profit with concrete multipliers': ParameterSpec(
                name='ATR stop/take-profit with concrete multipliers',
                min_val=1.0,
                max_val=6.0,
                default=1.5,
                param_type='float',
                step=None,
            ),
            'rsi_oversold': ParameterSpec(
                name='rsi_oversold',
                min_val=20,
                max_val=80,
                default=30,
                param_type='int',
                step=None,
            ),
            'ATR stop/take-profit multiplier': ParameterSpec(
                name='ATR stop/take-profit multiplier',
                min_val=0.75,
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
                step=None,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=2.0,
                default=2.5,
                param_type='float',
                step=None,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=4.0,
                param_type='float',
                step=None,
            ),
        }
    
    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        
        # Implement explicit LONG / SHORT / FLAT logic (if applicable)
        
        warmup = params['warmup']
        signals[:warmup] = 0.0
        return signals