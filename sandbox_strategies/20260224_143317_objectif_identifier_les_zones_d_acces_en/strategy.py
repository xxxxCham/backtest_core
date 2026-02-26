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
        return ['ema', 'obv', 'bollinger']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_min': 20,
         'donchian_bottom': 20,
         'ema_period': 10,
         'leverage': 1,
         'obv_top': 90,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50,
         'williamsr_fast': 10,
         'williamsr_slow': 3}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=5,
                max_val=50,
                default=10,
                param_type='int',
                step=1,
            ),
            'obv_top': ParameterSpec(
                name='obv_top',
                min_val=80,
                max_val=95,
                default=90,
                param_type='float',
                step=0.1,
            ),
            'donchian_bottom': ParameterSpec(
                name='donchian_bottom',
                min_val=20,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'adx_min': ParameterSpec(
                name='adx_min',
                min_val=15,
                max_val=60,
                default=20,
                param_type='int',
                step=1,
            ),
            'williamsr_fast': ParameterSpec(
                name='williamsr_fast',
                min_val=10,
                max_val=80,
                default=10,
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
        def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
                signals = pd.Series(0.0, index=df.index, dtype=np.float64)
                n = len(df)

                # Initialize long_mask and short_mask as boolean series of zeros with the same length as df
                long_mask = np.zeros(n, dtype=bool)
                short_mask = np.zeros(n, dtype=bool)

                # Implement explicit LONG / SHORT / FLAT logic
                warmup = int(params.get("warmup", 50))

                signals.iloc[:warmup] = 0.0

                return signals
        signals.iloc[:warmup] = 0.0
        return signals
