from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='My Awesome Strategy')
        
    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'macd', 'stochastic']
    
    @property
    def default_params(self) -> Dict[str, Any]:
        return {'fastd_length': 3,
                'leverage': 1,
                'period': 60,
                'slowk_length': 14,
                'stop_atr_mult': 1.5,
                'tp_atr_mult': 3.0,
                'warmup': 50}
    
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
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
    
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        # Generate signals for each indicator crossover and sentiment
        
        rsi = ta.momentum.RSIIndicator(arr['rsi'], period=14)
        macd = ta.trend.MACDHisto(arr['macd'])
        stochastic = ta.momentum.STOCH(arr['stochastic']['stoch_k'], arr['stochastic']['stoch_d'], window=14)
        
        crossover = np.where((rsi > indicators['macd']["macd"]), 1, 0)
        bullish_sentiment = np.where(stochastic['%K'] > stochastic['%D'], 1, 0)
    
        # Long signal
        if (crossover == 1 and bullish_sentiment == 1):
            signals[name] = 1.0
        
        else:
            signals[name] = -1.0
            
    return signals