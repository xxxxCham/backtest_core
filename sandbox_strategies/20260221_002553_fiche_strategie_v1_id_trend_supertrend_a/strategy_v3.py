from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='FICHE_SUPERTREND')

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
                min_val=0.25,
                max_val=4.5,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=3,
                max_val=6,
                default=4.5,
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
        def generate_signals(indicators, df, default_params={}, leverage=1):
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)

            # LONG intent
            if (close > indicators['supertrend']["supertrend"]['upper'] and indicators['adx']["adx"]['adx'] > 35 and atr >= risk_management.stop_atr_mult * threshold(atr) and direction == 1):
                signals[df.index] = 1.0

            # SHORT intent
            if (close < indicators['supertrend']["supertrend"]['lower'] and indicators['adx']["adx"]['adx'] > 35 and atr <= -risk_management.stop_atr_mult * threshold(-atr) and direction == -1):
                signals[df.index] = -1.0

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
