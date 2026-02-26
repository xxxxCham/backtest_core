from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='adausdc_mean_reversion_cci_keltner_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['cci', 'keltner', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'atr_min': 0.0005,
            'atr_period': 14,
            'cci_extreme': 200,
            'cci_period': 20,
            'keltner_atr_mult': 1.5,
            'keltner_period': 20,
            'leverage': 1,
            'stop_atr_mult': 1.1,
            'tp_atr_mult': 3.2,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'cci_period': ParameterSpec(
                name='cci_period',
                min_val=10,
                max_val=40,
                default=20,
                param_type='int',
                step=1,
            ),
            'cci_extreme': ParameterSpec(
                name='cci_extreme',
                min_val=100,
                max_val=300,
                default=200,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=7,
                max_val=28,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_min': ParameterSpec(
                name='atr_min',
                min_val=0.0001,
                max_val=0.005,
                default=0.0005,
                param_type='float',
                step=0.1,
            ),
            'keltner_period': ParameterSpec(
                name='keltner_period',
                min_val=10,
                max_val=40,
                default=20,
                param_type='int',
                step=1,
            ),
            'keltner_atr_mult': ParameterSpec(
                name='keltner_atr_mult',
                min_val=1.0,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.1,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=3.2,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=20,
                max_val=200,
                default=50,
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
        }

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        # Prepare output series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)

        # Extract parameters
        warmup = int(params.get('warmup', 50))
        cci_extreme = params.get('cci_extreme', 200)
        atr_min = params.get('atr_min', 0.0005)

        # Extract indicator arrays
        cci_vals = indicators['cci']            # numpy array
        atr_vals = indicators['atr']            # numpy array
        keltner_lower = indicators['keltner']['lower']
        keltner_upper = indicators['keltner']['upper']

        # Close price array
        close_arr = df['close'].values

        # Entry conditions
        long_mask = (close_arr < keltner_lower) & (cci_vals < -cci_extreme) & (atr_vals > atr_min)
        short_mask = (close_arr > keltner_upper) & (cci_vals > cci_extreme) & (atr_vals > atr_min)

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Zero out warmup period
        signals.iloc[:warmup] = 0.0

        return signals