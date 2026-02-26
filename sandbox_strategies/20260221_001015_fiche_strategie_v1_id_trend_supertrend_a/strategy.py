from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='FICHE_STRATEGY v1')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 2.5, 'tp_atr_mult': 5.5, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.1,
                max_val=3.0,
                default=2.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=0.67,
                max_val=8.0,
                default=5.5,
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
            # close values are numpy arrays (or dict of numpy arrays)
            close = df['close']

            for i, value in enumerate(close.values()):
                if 'supertrend' in indicators and \
                   'direction' in indicators['supertrend']:
                    supertrend_dir = indicators['supertrend']['direction'][i]

                    # LONG intent
                    if (supertrend_dir > 0 and \
                        adx[i]['adx'] > 25 or \
                        close[i] < -supertrend_dir and \
                        adx[i]['adx'] > 25):
                         signals[i] = 1.0
                    # SHORT intent
                    elif (supertrend_dir < 0 and \
                          adx[i]['adx'] < 10 or \
                          close[i] > supertrend_dir and \
                          adx[i]['adx'] < 10):
                         signals[i] = -1.0
                    else: # neither long nor short, use default value of 0
                        signals[i] = 0.0
                elif 'adx' in indicators:
                    # LONG intent
                    if (adx[i]['adx'] > 25 or \
                       close[i] < -indicators['donchian']['lower'][i-1]):
                         signals[i] = 1.0

                    # SHORT intent
                    elif (adx[i]['adx'] < 10 or \
                          close[i] > indicators['donchian']['upper'][i+1]):
                        signals[i] = -1.0
                else:
                    continue
        signals.iloc[:warmup] = 0.0
        return signals
