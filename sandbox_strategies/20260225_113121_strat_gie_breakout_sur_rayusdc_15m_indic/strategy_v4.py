from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='rayusdc_ema_aroon_volume_breakout_revamped')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'aroon', 'volume_oscillator', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'aroon_period': 14,
         'ema_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 2.0,
         'volume_sma_period': 20,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=5,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'aroon_period': ParameterSpec(
                name='aroon_period',
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
                default=2.0,
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
        ema_period = params.get('ema_period', 20)
        volume_sma_period = params.get('volume_sma_period', 20)
        
        ema_values = indicators['ema']
        aroon_up = indicators['aroon']["aroon_up"]
        aroon_down = indicators['aroon']["aroon_down"]
        volume_oscillator = indicators['volume_oscillator']
        
        long_condition = (
            (df['close'] > ema_values) & 
            (aroon_up > aroon_down) & 
            (volume_oscillator > np.mean(volume_oscillator, axis=0, keepdims=True))
        )

        short_condition = (
            (df['close'] < ema_values) & 
            (aroon_down > aroon_up) & 
            (volume_oscillator > np.mean(volume_oscillator, axis=0, keepdims=True))
        )

        signals[long_condition] = 1.0
        signals[short_condition] = -1.0
        signals.iloc[:warmup] = 0.0
        return signals