from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Snake Case Name')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'cci', 'donchian', 'macd']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'Cci_overbought': 90,
         'Cci_oversold': 10,
         'Macd_signal_cross_under': 0.5,
         'leverage': 2,
         'stop_atr_mult': 1.4,
         'tp_atr_mult': 3.2,
         'warmup': 75}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'Cci_overbought': ParameterSpec(
                name='Cci_overbought',
                min_val=-100,
                max_val=100,
                default=90,
                param_type='int',
                step=1,
            ),
            'Cci_oversold': ParameterSpec(
                name='Cci_oversold',
                min_val=-200,
                max_val=300,
                default=10,
                param_type='int',
                step=1,
            ),
            'Macd_signal_cross_under': ParameterSpec(
                name='Macd_signal_cross_under',
                min_val=0.5,
                max_val=1.5,
                default=0.5,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.3,
                max_val=2.4,
                default=1.4,
                param_type='float',
                step=0.1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=2,
                param_type='int',
                step=1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=3.2,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # implement explicit LONG / SHORT / FLAT logic
        signals.iloc[:warmup] = 0.0
        signals.iloc[:warmup] = 0.0
        return signals
