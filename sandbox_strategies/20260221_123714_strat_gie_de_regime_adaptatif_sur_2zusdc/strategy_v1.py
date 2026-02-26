from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Phase Locked Adaptive 2ZUSDC 15m')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'obv', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adxr_period': 14,
         'leverage': 1,
         'obv_numbars': 25,
         'rsi_period': 14,
         'stop_atr_mult': 1.3,
         'tp_atr_mult': 2.6,
         'warmup': 70}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'obv_numbars': ParameterSpec(
                name='obv_numbars',
                min_val=20,
                max_val=30,
                default=25,
                param_type='int',
                step=1,
            ),
            'adxr_period': ParameterSpec(
                name='adxr_period',
                min_val=7,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=2.0,
                default=1.3,
                param_type='float',
                step=0.1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=1,
                param_type='int',
                step=1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=2.6,
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
        def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params:Dict[str,Any]) -> np.ndarray:
            signals = np.zeros(len(df), dtype=np.float64) # Initialize signal series with zeros

            n = len(df) 
            long_mask = np.zeros(n, dtype=bool)  
            short_mask = np.zeros(n, dtype=bool)   

            warmup = params["warmup"] if "warmup" in params else 50 

            signals[ :warmup] = 0.0  

            # PHASE LOCK LOGIC HERE (IF APPLICABLE)

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
