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
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        # Generate ONLY the body lines to insert inside generate_signals. Do NOT generate class/imports/function signature or indicator values.

        indicators = {
            "rsi": None,
            "ema": None,
            "atr": None,
        }

        def generate_signals(df):

            # Check if momentum is high and risk controlled for LONG entry 
            long_momentum = df['close'].rolling('14').mean().shift(-7) > df['close'].rolling('20').mean()  
            long_risks = indicators['atr'] * 2.5 / (df['open'] - df['close']) # calculate risk for LONG entry

            signals_long = np.where((indicators['rsi'] is not None and indicators['rsi'].any()) & (indicators['ema'] is not None and indicators['ema'].any()) & 
                                    long_momentum)  

            # Assign '1' for LONG entry, '-1' for short exit and '0' otherwise.    
            signals_long = np.where((indicators['rsi'] is not None and indicators['rsi'][signals_long] > 50) & (indicators['ema'] is not None and indicators['ema'][signals_long][:3]>indicators['ema'][signals_long]) & 
                                    long_momentum,1,-1)    
            signals_long = np.where((np.abs(df['close'] - df['high']) < (0.2 * df['close'])) & ((df['low']-df['open'].shift()) > (0.5*df['open']), 1) ,1,-1)

            # Assign signals for LONG entry     
            signals_long[:len(signals_long)] = 1  
            signals[df.index] |= signals_long 

            # Check if momentum is low and risk controlled for SHORT entry 
            short_momentum = df['close'].rolling('14').mean().shift(-7) < df['close'].rolling('20').mean()  
            short_risks = indicators['atr'] * 2.5 / (df['open'] - df['close']) # calculate risk for SHORT entry    

            signals_short = np.where((indicators['rsi'] is not None and indicators['rsi'].any()) & (indicators['ema'] is not None and indicators['ema'].any()) & 
                                    short_momentum)  

            # Assign '-1' for SHORT entry, '1' for long exit and '0' otherwise.    
            signals_short = np.where((indicators['rsi'] is not None and indicators['rsi'][signals_short] < 50) & (indicators['ema'] is not None and indicators['ema'][signals_short][:3]<indicators['ema'][signals_short]) & 
                                    short_momentum,1,-1)    
            signals_short = np.where((np.abs(df['close'] - df['high']) > (0.2 * df['open'])) & ((df['low']-df['open'].shift()) < (-0.5*df['open']), 1) ,1,-1)

            # Assign signals for SHORT entry     
            signals_short[:len(signals_short)] = -1  
            signals[df.index] |= signals_short 

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
