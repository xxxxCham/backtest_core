from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='FICHE_STRATEGIE v1')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 20,
         'atrratio': 1.5,
         'atrstop': 0.7649385344827586,
         'leverage': 1,
         'ltrayv': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'tptrayr': 2.3299770689655133,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'atrratio': ParameterSpec(
                name='atrratio',
                min_val=1.0,
                max_val=3.0,
                default=1.5,
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
        def generate_signals(df):
            signals = np.nan # Initialize signal series with NaN values

            for index, row in df.iterrows():
                long_condition = (indicators['supertrend']['direction'] == 1) & ((adx > 30).all())
                short_condition = (indicators['supertrend']['direction'] == -1) & ((adx < 25).all())

                if any([long_condition, short_condition]): # If either long or short condition is True
                    df.at[index, 'signal'] = 1.0 # Assign signal of 1 for LONG and SHORT conditions

                else:
                    df.at[index, 'signal'] = -1.0 # Otherwise assign signal of -1

            return signals # Return generated signals as a Series object with the same index as input DataFrame
        signals.iloc[:warmup] = 0.0
        return signals
