from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_stochastic_williams_volume_adjusted')

    @property
    def required_indicators(self) -> List[str]:
        return ['stochastic', 'williams_r', 'volume_oscillator', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'stochastic_d_period': 3,
         'stochastic_k_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'volume_oscillator_long': 26,
         'volume_oscillator_short': 12,
         'warmup': 50,
         'williams_r_period': 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stochastic_k_period': ParameterSpec(
                name='stochastic_k_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'williams_r_period': ParameterSpec(
                name='williams_r_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=3.0,
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
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        stochastic_k = indicators['stochastic']["stoch_k"]
        williams_r = indicators['williams_r']
        volume_oscillator = indicators['volume_oscillator']
        
        long_condition = (
            (stochastic_k < 20) & 
            (stochastic_k > np.roll(stochastic_k, 1)) &
            (williams_r < -80) &
            (volume_oscillator > 0) &
            (volume_oscillator > np.mean(volume_oscillator[:5]))
        )

        short_condition = (
            (stochastic_k > 80) & 
            (stochastic_k < np.roll(stochastic_k, 1)) &
            (williams_r > -20) &
            (volume_oscillator < 0) &
            (volume_oscillator < np.mean(volume_oscillator[:5]))
        )

        signals[long_condition] = 1.0
        signals[short_condition] = -1.0
        signals.iloc[:warmup] = 0.0
        return signals