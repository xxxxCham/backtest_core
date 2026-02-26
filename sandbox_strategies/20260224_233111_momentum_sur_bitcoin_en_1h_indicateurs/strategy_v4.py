from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Phase Lock')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'macd']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'AroonUp_uptrend': 50,
         'RSI_overbought': 70,
         'RSI_oversold': 30,
         'SL': 2,
         'TP': 6,
         'VAPOR_TREND.ADU_downtrend': 25,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'RSI_oversold': ParameterSpec(
                name='RSI_oversold',
                min_val=10,
                max_val=90,
                default=30,
                param_type='int',
                step=1,
            ),
            'AroonUp_uptrend': ParameterSpec(
                name='AroonUp_uptrend',
                min_val=0,
                max_val=100,
                default=50,
                param_type='int',
                step=1,
            ),
            'VAPOR_TREND.ADU_downtrend': ParameterSpec(
                name='VAPOR_TREND.ADU_downtrend',
                min_val=0,
                max_val=100,
                default=25,
                param_type='int',
                step=1,
            ),
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
        # Implement your trading logic here
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Warm up protection
        signals.iloc[:warmup] = 0.0
        signals.iloc[:warmup] = 0.0
        return signals
