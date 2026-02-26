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
        return {'leverage': 1,
         'no_lookahead': True,
         'only_registry_indicators': True,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=1.75,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=6.0,
                default=2.5,
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
        def generate_signals(indicators):
            df = pd.DataFrame(...) # input dataframe goes here

            signals = pd.Series(0.0, index=df.index, dtype=np.float64)

            for i, row in df.iterrows():
                close = row['close']

                if (indicators['supertrend']['supertrend'][i] == 1 and \
                    indicators['adx']['adx'][i] > 25 and \
                    indicators['supertrend']['direction'][i] == -1 and \
                    indicators['adx']['adx'][i] > 25):
                     signals[i] = 1.0 # LONG intent met

                elif (indicators['supertrend']['supertrend'][i] == 1 and \
                       indicators['adx']['adx'][i] < 20 and \
                       indicators['supertrend']['direction'][i] == -1 and \
                       indicators['adx']['adx'][i] < 20):
                     signals[i] = -1.0 # SHORT intent met

                else:
                    signals[i] = 0.0  

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
