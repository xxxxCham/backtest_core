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
        return ['bollinger', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 2, 'rsi_period': 8, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 40}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=2,
                max_val=30,
                default=8,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.5,
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
        def generate_signals(indicators):
            # Indicators available in this method: ['bollinger', 'rsi', 'atr']

            # LONG intent: {'close < indicators['bollinger']['lower'] AND rsi < 50 AND adx > 25': 'ENTRY_LONG'} AND {'close > indicators['bollinger']['upper'] AND rsi > 70 AND adx > 25': 'ENTRY_SHORT}
            long_signals = []

            for i, row in df.iterrows():
                close = row['close']

                # Check if the conditions are met for LONG entry
                if any(row[key] < val[0] and row[key] > val[1] 
                       and row[indicator + '_adx'] > threshold_adx  
                       and row[indicator + '_plus_di'] >= abs(threshold_adi) 
                       for key, val in LONG.items()):
                    long_signals.append(True)
                else:
                    long_signals.append(False)

            signals['long'] = signals['long'].map(lambda x: -1 if x == True else x) # assign 1.0 to 'long' signal

            # SHORT intent: {'close > indicators['bollinger']['lower'] AND rsi < 60 AND adx > 25': 'ENTRY_LONG'} AND {'close < indicators['bollinger']['upper'] AND rsi > 70 AND adx > 25': 'ENTRY_SHORT}
            short_signals = []

            for i, row in df.iterrows():
                close = row['close']

                # Check if the conditions are met for SHORT entry
                if any(row[key] > val[0] and row[key] < val[1] 
                       and row[indicator + '_adx'] >= threshold_adi  
                       and row[indicator + '_plus_di'] <= abs(threshold_adi) 
                       for key, val in SHORT.items()):
                    short_signals.append(True)
                else:
                    short_signals.append(False)

            signals['short'] = signals['short'].map(lambda x: -1 if x == True else x) # assign 1.0 to 'long' signal

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
