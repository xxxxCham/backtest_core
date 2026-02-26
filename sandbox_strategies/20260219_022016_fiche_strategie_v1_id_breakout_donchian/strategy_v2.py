from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_donchian_adx')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'donchian', 'adx']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        def generate_signals(df):
            # Indicators available in this method: ['bollinger', 'donchian', 'adx']
            indicators = {
                'bollinger': ['upper', 'middle'],  # or 'lower' for bollinger band lower indicator
                'donchian': ['lower'],  # not necessary if using donchian channel with lower and upper
                'adx': ['adx', '+di', '-di']  
            }

            signals = pd.Series(np.zeros_like(df), index=df.index, dtype=np.float64)

            for i in df.index:

                # LONG signal
                if (df['close'][i] > np.mean(df[indicators['bollinger']['middle']][0]) and \
                    indicators['adx']['adx'] > 30):

                    signals[i] = 1.0   # buy signal, long position

                # SHORT signal
                elif (df['close'][i] < np.mean(df[indicators['donchian']['lower']][0]) and \
                      indicators['adx']['adx'] < 20):

                    signals[i] = -1.0   # sell signal, short position

                else:
                    signals[i] = 0.0   # no trading signal for this bar, keep the last trade decision unchanged

            return signals
        return signals
