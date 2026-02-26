from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='FICHE_STRATEGIE')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
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
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=1,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        def generate_signals(df, default_params):
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)

            # LONG intent
            long_intent_bollinger = df['close'] >= df[indicators['bollinger']['upper']]
            long_intent_rsi = df[indicators['rsi']] <= 25
            long_entry_logic_bollinger = df['close'] < df[indicators['bollinger']['lower']]

            # SHORT intent
            short_intent_bollinger = df['close'] <= df[indicators['bollinger']['upper']]
            short_intent_rsi = df[indicators['rsi']] >= 80
            short_entry_logic_bollinger = df['close'] > df[indicators['bollinger']['lower']]

            # Combining both intentions
            signals[long_intent_bollinger & long_intent_rsi] = 1.0
            signals[(~short_intent_bollinger) | (~short_intent_rsi)] = -1.0
            signals['bb_stop_long'] = df['close'] + default_params[indicators['adx']['plus_di'], 'atr'] * (default_params[indicators['adx']['minus_di'], 'atr']) / 2
            signals['bb_tp_long'] = df['close'] - df['close']

            return signals
        return signals
