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
        return ['bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 2,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 20}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ATR': ParameterSpec(
                name='ATR',
                min_val=0.5,
                max_val=8.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'stop_mult': ParameterSpec(
                name='stop_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
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
        def generate_signals(indicators, default_params):
            df = pd.DataFrame(default_params)   # Assuming 'df' is a DataFrame object in this context

            signals = pd.Series(np.zeros(len(df), dtype=np.float64)) 

            for i, row in df.iterrows():
                close = row['close']

                if indicators['bollinger']['upper'][i] < close and \
                   (indicators['atr'][i] < indicators['atr'][i]['30day']) and \
                   adx[i][2] > 25:   # Assuming 'adx' is an array of ADX values for each bar
                    signals[i] = -1.0
                elif indicators['bollinger']['lower'][i] > close and \
                      (indicators['atr'][i]['30day'] > indicators['atr'][i]) and \
                      adx[i][2] < 25:   # Assuming 'adx' is an array of ADX values for each bar
                    signals[i] = 1.0
                elif stochastic[i, :] > 80 or rsi[i, :] > 70 and \
                      (indicators['atr'][i]['30day'] < indicators['atr'][i]) and \
                      adx[i][2] >= 50:   # Assuming 'stochastic' and 'rsi' are arrays of values for each bar.
                    signals[i] = -1.0
                elif stochastic[i, :] < 20 or rsi[i, :] < 30 and \
                      (indicators['atr'][i]['30day'] > indicators['atr'][i]) and \
                      adx[i][2] <= 50:   # Assuming 'stochastic' and 'rsi' are arrays of values for each bar.
                    signals[i] = 1.0
            return signals
        return signals
